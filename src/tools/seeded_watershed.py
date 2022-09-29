import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from skimage.filters import gaussian
from skimage.measure import label
from skimage.morphology import local_minima
import pyclesperanto_prototype as cle
from skimage.segmentation import watershed, relabel_sequential
from skimage.measure import regionprops
from tifffile import imwrite, imread
from tqdm import tqdm

from src.config.config import combine_cfgs
from src.data.preprocess_utils import load_files, erode_labels

# Load the .ENV path.
if find_dotenv() != "":
    load_dotenv(find_dotenv(), verbose=True)
    if os.getenv("PATH_DATA_TEST") is not None:
        PATH_DATA_TEST = Path(os.getenv("PATH_DATA_TEST"))


def seeded_watershed(image, mask, seeds=None, star=None, outline_sigma=0):

    image = np.asarray(image)

    if outline_sigma != 0:
        outline_blurred = gaussian(image, sigma=outline_sigma)
    else:
        outline_blurred = image

    if seeds is not None:
        return watershed(outline_blurred, markers=seeds, mask=mask)

    elif star is not None:
        # erode star prediction a bit because in some places it was going out of the cell membrane
        seeds_star = erode_labels(labels=[star], erosion_iterations=4, return_final_eroded_list=True)[0]

        return watershed(outline_blurred, markers=seeds_star, mask=mask)


def custom_seeds_watershed(image, mask, star=None, centroids_as_seeds=False, df=None, outline_sigma=0):

    if centroids_as_seeds:
        seeds = [(z, y, x) for z, y, x in zip(df["centroid_z"], df["centroid_y"], df["centroid_x"])]
        coords = np.asarray(seeds)
        coords_int = np.round(coords).astype(int)
        markers_raw = np.zeros_like(image)
        markers_raw[tuple(coords_int.T)] = 1 + np.arange(len(coords))

        wt = seeded_watershed(image, seeds=markers_raw, mask=mask, outline_sigma=outline_sigma)
    else:
        wt = seeded_watershed(image, mask=mask, star=star, outline_sigma=outline_sigma)

    return cle.smooth_labels(wt, None, 4.0)


def local_minima_seeded_watershed(image, spot_sigma, outline_sigma):
    """
    Function from napari_segment_blobs_and_things_with_membranes by Robert Haase. See [1]

    Segment cells in images with fluorescently marked membranes.
    The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, local minima detection and a seeded watershed. See also [2]
    --------
    [1] https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes
    [2] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    """

    image = np.asarray(image)

    spot_blurred = gaussian(image, sigma=spot_sigma)

    spots = label(local_minima(spot_blurred))

    if outline_sigma == spot_sigma:
        outline_blurred = spot_blurred
    else:
        outline_blurred = gaussian(image, sigma=outline_sigma)

    return watershed(outline_blurred, spots)


def thresholded_local_minima_seeded_watershed(image, spot_sigma, outline_sigma, minimum_intensity):
    """
    Function from napari_segment_blobs_and_things_with_membranes by Robert Haase. See [1]

    Segments cells in images with marked membranes that have a high signal intensity.
    The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, local minima detection and a seeded watershed.
    Afterwards, all objects are removed that have an average intensity below a given minimum_intensity.

    [1] https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes
    """
    labels = local_minima_seeded_watershed(image, spot_sigma=spot_sigma, outline_sigma=outline_sigma)

    # measure intensities
    stats = regionprops(labels, image)
    intensities = [r.mean_intensity for r in stats]

    # filter labels with low intensity
    new_label_indices, _, _ = relabel_sequential((np.asarray(intensities) > minimum_intensity) * np.arange(labels.max()))
    new_label_indices = np.insert(new_label_indices, 0, 0)
    new_labels = np.take(np.asarray(new_label_indices, np.uint32), labels)

    return new_labels


@click.command()
@click.option('-cfg', '--cfg_path', default=None, type=click.Path(exists=True), required=True,
              help="Path to the YAML configuration file of the new experiment, which will be used "
                   "to overwrite default behaviour.")
@click.argument('images_dir', envvar='PATH_DATA_TEST', type=click.Path(exists=True), required=False)
@click.argument('results_dir', type=click.Path(exists=False), required=False)
@click.argument('props_dir', type=click.Path(exists=True), required=False)
@click.argument('masks_dir', type=click.Path(exists=True), required=False)
def process_list_of_images(cfg_path, images_dir, centroids_as_seeds=False, results_dir=None, props_dir=None,
                           masks_dir=None, star_dir=None):
    """
    Parameters
    -----------
    cfg_path : str
        path to YAML config file
    images_dir : str
        directory containing images to be processed
    centroids_as_seeds : bool
        whether to use extracted centroids or direct prediction of StarDist is seeds for watershed
    results_dir : str
        directory where results will be saved. If not given determined by /base_project_dir/experiments/expX/results
    props_dir : str
        directory containing .csv files with measured region properties for each image (must be named same as images)
    masks_dir : str
        directory containing binary masks files. Where mask == False, pixels will not be segmented. Only needed when
        custom seeded watershed is used
    star_dir : str
        directory containing StarDist predicted labels that will be used to extract centroids or directly as seeds
    Returns
    ---------
    None
    """
    cfg = combine_cfgs(cfg_path)

    if images_dir is None:
        images_dir = cfg.MODEL.PREDICTION.TEST_DIR

    base_dir = Path(__file__).resolve().parents[2]  # get directory of the project
    exp = os.path.split(cfg_path)[1].split('.')[0]  # get experiment name e.g. "exp06"
    exp_dir = os.path.join(base_dir, "experiments", exp)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    if results_dir is None:
        results_dir = os.path.join(exp_dir, "results")

    if props_dir is not None:
        cfg.WATERSHED.PROPS_DIR = props_dir
    if masks_dir is not None:
        cfg.WATERSHED.MASKS_DIR = masks_dir

    images, fnames = load_files(images_dir, step=1, return_fnames=True)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for img, name in tqdm(zip(images, fnames), total=len(images), desc="Segmenting images"):

        if cfg.WATERSHED.CUSTOM_SEEDED_WT is False:
            labeled = thresholded_local_minima_seeded_watershed(img, spot_sigma=cfg.WATERSHED.SPOT_SIGMA,
                                                                outline_sigma=cfg.WATERSHED.OUTLINE_SIGMA,
                                                                minimum_intensity=cfg.WATERSHED.MIN_INTENSITY)
            labeled = cle.smooth_labels(labeled, None, 4.0)
        else:
            assert cfg.WATERSHED.PROPS_DIR is not None or cfg.WATERSHED.PROPS_DIR != "", "Reg props directory not given"
            assert cfg.WATERSHED.MASKS_DIR is not None or cfg.WATERSHED.MASKS_DIR != "", "Masks directory not given"

            # mask = imread(os.path.join(cfg.WATERSHED.MASKS_DIR, name + "_probabilities.tif"))
            mask = imread(os.path.join(cfg.WATERSHED.MASKS_DIR, name))

            if centroids_as_seeds:

                df = pd.read_csv(os.path.join(cfg.WATERSHED.PROPS_DIR, name.split(".")[0] + ".csv"))
                df.dropna(inplace=True, subset=["centroid_x", "centroid_y", "centroid_z"])

                labeled = custom_seeds_watershed(img, mask=mask, centroids_as_seeds=centroids_as_seeds,
                                                 outline_sigma=cfg.WATERSHED.OUTLINE_SIGMA, df=df)

            else:
                star = imread(os.path.join(star_dir, name))

                labeled = custom_seeds_watershed(img, mask=mask, centroids_as_seeds=centroids_as_seeds,
                                                 outline_sigma=cfg.WATERSHED.OUTLINE_SIGMA, star=star)

        imwrite(os.path.join(results_dir, name), labeled)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
