import logging
import math
from pathlib import Path

from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from stardist.matching import group_matching_labels
from tifffile import imread, imwrite
from sklearn.metrics import jaccard_score
import os
import click

import pandas as pd
import pyclesperanto_prototype as cle
import numpy as np
from dotenv import find_dotenv, load_dotenv
import surface_distance
from tqdm import tqdm

from src.data.preprocess_utils import load_files

# todo use dask to avoid memory errors

# Load the .ENV path if it exists
if find_dotenv() != "":
    load_dotenv(find_dotenv(), verbose=True)
    if os.getenv("PATH_DATA_RAW") is not None:
        PATH_DATA_RAW = Path(os.getenv("PATH_DATA_RAW"))
    if os.getenv("PATH_LABELS") is not None:
        PATH_LABELS = Path(os.getenv("PATH_LABELS"))


@click.command()
@click.argument('images_dir', envvar='PATH_DATA_RAW', type=click.Path(exists=True), required=True)
@click.argument('labels_dir', envvar='PATH_LABELS', type=click.Path(exists=True), required=True)
@click.argument('gt_dir', type=click.Path(exists=True), required=False)
@click.argument('results_dir', required=True)
def get_measurements(images_dir, labels_dir, gt_dir=None, remove_bg_label=False, results_dir=None, spacing=None):
    cle.select_device("RTX")
    logger = logging.getLogger(__name__)
    logger.info(f"Images from:{images_dir} will be measured.")

    reg_props = get_regprops_timelapse(intensity_image_dir=images_dir, label_image_dir=labels_dir, gt_dir=gt_dir,
                                       remove_bg_label=remove_bg_label,
                                       region_props_source="neighborhood", results_dir=results_dir,
                                       spacing=spacing)

    df = pd.DataFrame(reg_props)

    df.to_csv(os.path.join(results_dir, "all.csv"), index=False)


def get_regprops_timelapse(
        intensity_image_dir,
        label_image_dir,
        gt_dir=None,
        remove_bg_label=False,
        region_props_source="neighborhood",
        n_closest_points_list=None,
        results_dir=None,
        spacing=None
) -> pd.DataFrame:
    """
    Calculate Region properties of the timelapse with additional parameters as sphericity and surface area, and also
    with IoU measurements according to the ground truth.
    """
    if n_closest_points_list is None:
        n_closest_points_list = [2, 3, 4]
    else:
        n_closest_points_list = list(n_closest_points_list)

    # and select columns, depending on if intensities, neighborhood
    # and/or shape were selected
    columns = ["label", "centroid_x", "centroid_y", "centroid_z"]
    intensity_columns = [
        "min_intensity",
        "max_intensity",
        "sum_intensity",
        "mean_intensity",
        "standard_deviation_intensity",
    ]
    shape_columns = [
        "area",
        "volume",
        "surf_area",
        "sphericity",
        "mean_distance_to_centroid",
        "max_distance_to_centroid",
        "mean_max_distance_to_centroid_ratio",
    ]
    if gt_dir is not None:
        columns += ["IoU"]
    if "intensity" in region_props_source:
        columns += intensity_columns
    if "shape" in region_props_source:
        columns += shape_columns
    if "neighborhood" in region_props_source:
        columns += shape_columns
        columns += intensity_columns

    if spacing is None:
        spacing = (0.001, 0.001, 0.001)

    # intensity_images = load_files(intensity_image_dir)    don´t load all because of memory problems
    # label_images, fnames = load_files(label_image_dir, return_fnames=True)
    if gt_dir is not None:
        gt_images = load_files(gt_dir)

    fnames_lbl = load_files(label_image_dir, return_only_fnames=True)
    fnames_raw = load_files(intensity_image_dir, return_only_fnames=True)

    reg_props_all = []

    Path(os.path.join(label_image_dir, "matched")).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for t, (name_lbl, name_raw) in tqdm(enumerate(zip(fnames_lbl, fnames_raw)), total=len(fnames_raw)):

        intensity_image = imread(os.path.join(intensity_image_dir, name_raw))
        label_image = imread(os.path.join(label_image_dir, name_lbl))

        if remove_bg_label:
            df = pd.DataFrame(regionprops_table(np.array(label_image).astype("int32"), properties=["label", "area"]))
            bg = df[(df['area'] == df['area'].max())]['label'].values[0]

            def _remove_bg(j):
                if j == bg:
                    return 0
                else:
                    return j

            func_mask = np.vectorize(_remove_bg)
            y_pred = func_mask(label_image)
        else:
            y_pred = label_image

        y_pred, _, _ = relabel_sequential(np.array(y_pred).astype("int32"))
        if gt_dir is not None:
            y_true, y_pred = group_matching_labels([np.array(gt_images[t]).astype("int32"), np.array(y_pred).astype("int32")],
                                                   thresh=0.3)  # match so labels of same object would have same index

        imwrite(os.path.join(label_image_dir, "matched", name_raw), y_pred)

        all_reg_props_single_t = pd.DataFrame(
            cle.statistics_of_labelled_pixels(
                intensity_image, y_pred
            )
        )
        if gt_dir is not None:
            iou_per_lbl = []
            for i in range(1, y_pred.max() + 1):
                label_id = i
                only_current_label_id_y_true = np.where(y_true == label_id, 1, 0)
                only_current_label_id_y_pred = np.where(y_pred == label_id, 1, 0)
                iou_per_lbl.append(jaccard_score(np.array(only_current_label_id_y_true).flatten(),
                                                 np.array(only_current_label_id_y_pred).flatten()))
            iou = pd.DataFrame(iou_per_lbl, columns=["IoU"])

        surf_distances = []
        for i in range(1, y_pred.max() + 1):
            label_id = i
            only_current_label_id = np.where(y_pred == label_id, True, False)

            surf_distances.append(surface_distance.compute_surface_distances(np.array(only_current_label_id),
                                                                             np.array(only_current_label_id),
                                                                             spacing_mm=spacing))

        surf_area = pd.DataFrame(
            [sum(a) * 1000000 for a in pd.DataFrame(surf_distances)["surfel_areas_pred"]], columns=["surf_area"])  # converting from sqrt mm t sqrt microns

        volume = all_reg_props_single_t["area"] * 0.173 ** 3
        all_reg_props_single_t["volume"] = volume

        s = []
        for a, v in zip(surf_area["surf_area"], all_reg_props_single_t["volume"]):
            try:
                s.append(((36 * math.pi * v ** 2) ** (1 / 3)) / a)
            except ZeroDivisionError:
                s.append(0)

        sphericity = pd.DataFrame(s, columns=["sphericity"])

        if gt_dir is not None:
            all_reg_props_single_t = pd.concat([all_reg_props_single_t, surf_area, sphericity, iou], axis=1)
        else:
            all_reg_props_single_t = pd.concat([all_reg_props_single_t, surf_area, sphericity], axis=1)

        if "neighborhood" in region_props_source:
            reg_props_single_t = region_props_with_neighborhood_data(
                y_pred,
                n_closest_points_list,
                all_reg_props_single_t[columns],
            )
        else:
            reg_props_single_t = all_reg_props_single_t[columns]

        timepoint_column = pd.DataFrame({"Frame": np.full(len(reg_props_single_t), t)})
        actual_timepoint_column = pd.DataFrame({"actual_frame": np.full(len(reg_props_single_t), name_raw.split(".")[0])})
        reg_props_with_tp_column = pd.concat([reg_props_single_t, timepoint_column, actual_timepoint_column], axis=1)
        reg_props_all.append(reg_props_with_tp_column)

        df = pd.DataFrame(reg_props_with_tp_column)

        if results_dir is not None:
            df.to_csv(os.path.join(results_dir, name_raw.split(".")[0] + ".csv"), index=False)

    reg_props = pd.concat(reg_props_all)
    return reg_props


def region_props_with_neighborhood_data(
        label_image, n_closest_points_list: list, reg_props: pd.DataFrame
) -> pd.DataFrame:
    """
    From napari-clusters-plotter. See [1]

    Calculate neighborhood region properties and combine with other region properties

    Parameters
    ----------
    label_image : numpy array or dask array
        segmented image with background = 0 and labels >= 1
    reg_props: Dataframe
        region properties to be combined with
    n_closest_points_list: list
        number of closest neighbors for which neighborhood properties will be calculated

    [1] https://github.com/BiAPoL/napari-clusters-plotter
    """
    neighborhood_properties = {}

    # get the lowest label index to adjust sizes of measurement arrays
    min_label = int(np.min(label_image[np.nonzero(label_image)]))

    # determine neighbors of cells
    touch_matrix = cle.generate_touch_matrix(label_image)

    # ignore touching the background
    cle.set_column(touch_matrix, 0, 0)
    cle.set_row(touch_matrix, 0, 0)

    # determine distances of all cells to all cells
    pointlist = cle.centroids_of_labels(label_image)

    # generate a distance matrix
    distance_matrix = cle.generate_distance_matrix(pointlist, pointlist)

    # detect touching neighbor count
    touching_neighbor_count = cle.count_touching_neighbors(touch_matrix)
    cle.set_column(touching_neighbor_count, 0, 0)

    # conversion and editing of the distance matrix, so that it does not break cle.average_distance
    view_dist_mat = cle.pull(distance_matrix)
    temp_dist_mat = np.delete(view_dist_mat, range(min_label), axis=0)
    edited_dist_mat = np.delete(temp_dist_mat, range(min_label), axis=1)

    # iterating over different neighbor numbers for average neighbor distance calculation
    for i in n_closest_points_list:
        distance_of_n_closest_points = cle.pull(
            cle.average_distance_of_n_closest_points(cle.push(edited_dist_mat), n=i)
        )[0]

        # addition to the regionprops dictionary

        neighborhood_properties[
            f"avg distance of {i} closest points"
        ] = distance_of_n_closest_points

    # processing touching neighbor count for addition to regionprops (deletion of background & not used labels)
    touching_neighbor_c = cle.pull(touching_neighbor_count)
    touching_neighbor_count_formatted = np.delete(
        touching_neighbor_c, list(range(min_label))
    )

    # addition to the regionprops dictionary
    neighborhood_properties[
        "touching neighbor count"
    ] = touching_neighbor_count_formatted
    return pd.concat([reg_props, pd.DataFrame(neighborhood_properties)], axis=1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

