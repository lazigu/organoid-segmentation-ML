import os
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imwrite
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.filters import threshold_mean
from stardist.matching import group_matching_labels, relabel_sequential
from skimage.measure import regionprops_table
import pyclesperanto_prototype as cle

from src.data.preprocess_utils import load_dataset, erode_labels, load_files, dilate_labels
import logging


def postprocess(segm_path, masks_dir, exp_dir=None):
    # masks = erode_labels(masks_dir, erosion_iterations=3)

    # lbl_images, fnames = load_dataset(images_path, return_fnames=True)[0]
    (lbl_images, _), (masks, fnames) = load_dataset(segm_path, masks_dir, return_fnames=True)

    logger = logging.getLogger(__name__)

    save_path = os.path.join(exp_dir, "results")
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Postprocessed lbl_images will be saved to: {save_path}")

    finished = []
    # print("Matching")
    # matched = group_matching_labels(lbl_images)

    for img, mask, fname in tqdm(zip(lbl_images, masks, fnames), total=len(lbl_images),
                                 desc="Masking with UNET results"):
        img = img.astype("int32")
        # img = matched[i, :, :, :]
        # we make the edges of the mask a bit nicer because unet model is not very good since downscaled data is used for it
        mask = gaussian(mask, sigma=3)
        thresh = threshold_mean(mask)
        mask = mask > thresh
        # remove = set()

        df = pd.DataFrame(regionprops_table(img, properties=["label", "area"]))
        bg = df[(df['area'] == df['area'].max())]['label'].values[0]
        print(f"Background: {bg}")

        # def _check(i, j):
        #     if j == 0:
        #         return i
        #     else:
        #         return 0

        def _mask(i, j):
            if j == 0 or i == bg:
                return 0
            else:
                return i

        func_mask = np.vectorize(_mask)
        postprocessed_img = func_mask(img, mask)

        # take 1st item bc it returns a list
        # eroded = erode_labels(labels=[postprocessed_img], erosion_iterations=3)[0]
        # dilated = dilate_labels(labels=[eroded], fnames=[fname], dilate_iterations=6)[0]

        # expanded = expand_labels(eroded, distance=3)

        match = group_matching_labels(postprocessed_img)
        relabeled, _, _ = relabel_sequential(match)

        finished.append(match)

    print("Grouping")
    matched = group_matching_labels(finished, thresh=0.5)
    print(f"Matched shape: {matched.shape}")
    for i, fname in tqdm(zip(range(matched.shape[0]), fnames), desc=f"Saving to {save_path}", total=len(fnames)):
        img = matched[i, :, :, :]
        imwrite(os.path.join(save_path, fname), img)


def merge_to_most_dominant_label(lbl_path=None, labels=None, min_area=1500, save_dir=None):
    """
    Partially based on:
    https://stackoverflow.com/questions/72452267/finding-identity-of-touching-labels-objects-masks-in-images-using-python/72456945#72456945
    """
    from collections import Counter
    from skimage.measure import regionprops
    from skimage.graph import pixel_graph
    from collections import defaultdict

    logger = logging.getLogger(__name__)

    if lbl_path is not None:
        logger.info(f"Processing labels from: {lbl_path}")
        labels, fnames = load_files(files_dir=lbl_path, return_fnames=True)
    else:
        fnames = [str(i) + ".tif" for i in enumerate(labels)]

    if save_dir is None:
        save_dir = os.path.join(lbl_path, "postprocessed")

    Path(os.path.join(save_dir)).mkdir(parents=True, exist_ok=True)

    for label, fname in tqdm(zip(labels, fnames), total=len(labels), desc="Processing labels"):

        g, nodes = pixel_graph(
            label,
            mask=label.astype(bool),
            connectivity=3,  # count diagonals in 3D
        )

        g.eliminate_zeros()

        coo = g.tocoo()
        center_coords = nodes[coo.row]
        neighbor_coords = nodes[coo.col]

        center_values = label.ravel()[center_coords]
        neighbor_values = label.ravel()[neighbor_coords]

        pairs = defaultdict(list)

        for i, j in zip(center_values, neighbor_values):
            pairs[i].append(j)

        reg_props = regionprops(label)

        labels_to_merge = []

        for prop in reg_props:
            if prop["area"] < min_area:
                labels_to_merge.append(prop["label"])

        merge_to = {}

        for i in labels_to_merge:
            lst = list(map(int, pairs[int(i)]))
            if i in lst:
                lst = list(filter((i).__ne__, lst))
            print(
                f"Label: {i}, {Counter(lst)}, will be merged with (99999 means bg): "
                f"{max(lst, key=lst.count) if len(lst) > 0 else 99999}")  # just need very big id label, which will indicate it should be merged with a background
            merge_to[i] = max(lst, key=lst.count) if len(
                lst) > 0 else 99999  # empty Counter means that it has no neighbours and the label is an artifact and should be removed

        print(f"Labels that will be merged: {merge_to}")

        def _merge(i, j):
            if j == 99999:
                return 0
            elif j != 0:
                return j
            else:
                return i

        func_merge = np.vectorize(_merge)

        processed = np.copy(label)

        for (k, v) in merge_to.items():
            only_label_to_be_removed = np.where(processed == k, v, 0)
            processed = func_merge(processed, only_label_to_be_removed)

        # print(f"Final shape: {processed.shape}")

        if save_dir is not None:
            imwrite(os.path.join(save_dir, fname), processed)


def remove_second_object(lbl_path, save_dir=None):
    """
    Remove smaller label from the binary image with not touching objects, leaving only the biggest one.
    """
    from skimage import measure

    masks, fnames = load_files(lbl_path, return_fnames=True)

    if save_dir is None:
        save_dir = os.path.join(lbl_path, "removed_2nd")

    def _remove(i):
        if i == keep:
            return 1
        else:
            return 0

    func_mask = np.vectorize(_remove)

    Path(os.path.join(save_dir)).mkdir(parents=True, exist_ok=True)

    for mask, fname in tqdm(zip(masks, fnames), desc=f"Saving to {save_dir}", total=len(fnames)):
        # eroded = erode_labels(labels=[mask.astype("int32")], erosion_iterations=4)[0]
        # labeled = measure.label(eroded)
        # dilated = dilate_labels(labels=[labeled], dilate_iterations=4)[0]
        labeled = measure.label(mask)
        df = pd.DataFrame(regionprops_table(labeled, properties=["label", "area"]))
        keep = df[(df['area'] == df['area'].max())]['label'].values[0]

        processed = func_mask(labeled)

        imwrite(os.path.join(save_dir, fname), processed)


def get_masks_from_star_pmaps(masks_dir, save_dir=None):
    masks, fnames = load_files(masks_dir, return_fnames=True)

    if save_dir is None:
        save_dir = os.path.join(masks_dir, "masks")

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for mask, name in tqdm(zip(masks, fnames), total=len(fnames)):
        image1_t = cle.greater_constant(mask, None, 0.02999999999999925)
        image2_cl = cle.closing_labels(image1_t, None, 2.1)
        image2_el = cle.erode_labels(image2_cl, None, 2.0, False)

        imwrite(os.path.join(save_dir, name), image2_el)


def smooth_list_of_labels_from_path(labels_dir, save_dir=None):
    images, fnames = load_files(labels_dir, return_fnames=True)

    if save_dir is None:
        save_dir = os.path.join(labels_dir, "smoothed")

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for img, n in tqdm(zip(images, fnames), total=len(fnames)):
        processed = cle.smooth_labels(cle.smooth_labels(img, None, 4.0))
        imwrite(os.path.join(save_dir, n), processed)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
