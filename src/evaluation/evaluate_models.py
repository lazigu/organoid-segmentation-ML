import os

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential
from sklearn.metrics import brier_score_loss
from stardist.matching import matching_dataset, group_matching_labels
import surface_distance
import math
import statistics
from sklearn.metrics import log_loss
from statistics import mean
from tifffile import imread
from tqdm import tqdm


def evaluate(y_true, y_pred):
    """
    Function from Stardist. See https://github.com/stardist/stardist

    Help on function matching in module stardist.matching:

    matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False)
    Calculate detection/instance segmentation metrics between ground truth and predicted label images.

    Currently, the following metrics are implemented:

    'fp', 'tp', 'fn', 'precision', 'recall', 'accuracy', 'f1', 'criterion', 'thresh', 'n_true', 'n_pred',
    'mean_true_score', 'mean_matched_score', 'panoptic_quality'

    Corresponding objects of y_true and y_pred are counted as true positives (tp), false positives (fp), and false
    negatives (fn) whether their intersection over union (IoU) >= thresh (for criterion='iou', which can be changed)

    * mean_matched_score is the mean IoUs of matched true positives

    * mean_true_score is the mean IoUs of matched true positives but normalized by the total number of GT objects

    * panoptic_quality defined as in Eq. 1 of Kirillov et al. "Panoptic Segmentation", CVPR 2019

    Parameters
    ----------
    y_true: ndarray
        ground truth label image (integer valued)
    y_pred: ndarray
        predicted label image (integer valued)

    Returns
    -------
    A list of matching objects with different metrics as attributes

    """

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(y_true, y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]
    return stats


def get_boundary_based_metrics_per_object(predictions, ground_truth, spacing_mm=None, hausdorff_percent=95.0,
                                          tolerance_mm=0.000173 * 3):

    if spacing_mm is None:
        spacing_mm = (1, 1, 1)

    surface_distances = []
    vol_dice = []

    for y_pred, y_true in tqdm(zip(predictions, ground_truth), total=len(predictions),
                               desc="Computing surface distances dict for objects in each image"):

        y_pred, _, _ = relabel_sequential(np.array(y_pred).astype("int32"))
        y_true, y_pred = group_matching_labels([np.array(y_true).astype("int32"), np.array(y_pred).astype("int32")],
                                               thresh=0.3)  # match so labels of same object would have same index

        for i in range(1, (y_true.astype("int32")).max() + 1):
            label_id = i
            only_current_label_id_y_true = np.where(y_true == label_id, True, False)
            only_current_label_id_y_pred = np.where(y_pred == label_id, True, False)

            surface_distances.append(
                surface_distance.compute_surface_distances(only_current_label_id_y_true, only_current_label_id_y_pred,
                                                           spacing_mm=spacing_mm))

            vol_dice.append(surface_distance.compute_dice_coefficient(only_current_label_id_y_true.astype("int32"),
                                                                      only_current_label_id_y_pred.astype("int32")))

    avrg_surface_distances = [surface_distance.compute_average_surface_distance(d) for d in surface_distances]

    avrg_d_from_gt_to_pred = [f[0] * 1000 for f in avrg_surface_distances]
    avrg_d_from_pred_to_gt = [f[1] * 1000 for f in avrg_surface_distances]

    df = pd.concat([pd.DataFrame(avrg_d_from_gt_to_pred, columns=["avrg_d_from_gt_to_pred"]),
                    pd.DataFrame(avrg_d_from_pred_to_gt, columns=["avrg_d_from_pred_to_gt"])], axis=1)

    hausdorff_distances_all = [surface_distance.compute_robust_hausdorff(d, hausdorff_percent) * 1000 for d in surface_distances]

    df = pd.concat([df, pd.DataFrame(hausdorff_distances_all, columns=["HD95"])], axis=1)

    surface_overlap = [surface_distance.compute_surface_overlap_at_tolerance(d, tolerance_mm=tolerance_mm) for d in
                       surface_distances]  # tolerance - 0.000173*3 microns, i.e. 1 pixels?

    surf_overlap_of_gt_surf_w_pred_surf = [f[0] for f in surface_overlap]
    surf_overlap_of_pred_surf_w_gt_surf = [f[1] for f in surface_overlap]

    df = pd.concat(
        [df, pd.DataFrame(surf_overlap_of_gt_surf_w_pred_surf, columns=["surf_overlap_of_gt_surf_w_pred_surf"]),
         pd.DataFrame(surf_overlap_of_pred_surf_w_gt_surf, columns=["surf_overlap_of_pred_surf_w_gt_surf"])], axis=1)

    surface_dice = [surface_distance.compute_surface_dice_at_tolerance(d, tolerance_mm) for d in surface_distances]

    df = pd.concat([df, pd.DataFrame(surface_dice, columns=["DSCsurf"])], axis=1)
    df = pd.concat([df, pd.DataFrame(vol_dice, columns=["DSCvol"])], axis=1)

    return df


def get_boundary_based_metrics_per_frame(predictions, ground_truth, spacing_mm=None, hausdorff_percent=95.0,
                                         tolerance_mm=0.000173 * 3):
    returned = {}

    if spacing_mm is None:
        spacing_mm = (1, 1, 1)

    surface_distances = []

    for y_pred, y_true in tqdm(zip(predictions, ground_truth), total=len(predictions),
                               desc="Computing surface distances dict for each label"):

        y_pred, _, _ = relabel_sequential(np.array(y_pred).astype("int32"))
        y_true, y_pred = group_matching_labels([np.array(y_true).astype("int32"), np.array(y_pred).astype("int32")],
                                               thresh=0.3)  # match so labels of same object would have same index

        d_per_frame = []

        for i in range(1, (y_true.astype("int32")).max() + 1):
            label_id = i
            only_current_label_id_y_true = np.where(y_true == label_id, True, False)
            only_current_label_id_y_pred = np.where(y_pred == label_id, True, False)

            d_per_frame.append(
                surface_distance.compute_surface_distances(only_current_label_id_y_true, only_current_label_id_y_pred,
                                                           spacing_mm=spacing_mm))

        surface_distances.append(d_per_frame)

    avrg_surface_distances_all = []

    for surf_per_img in tqdm(surface_distances, desc="Computing average surface distances"):
        avrg_surface_distances = [surface_distance.compute_average_surface_distance(d) for d in surf_per_img]

        avrg_d_from_gt_to_pred = np.mean(
            np.array([f[0] for f in avrg_surface_distances if f[0] < float('inf') and not math.isnan(f[0])]))
        avrg_d_from_pred_to_gt = np.mean(
            np.array([f[1] for f in avrg_surface_distances if f[1] < float('inf') and not math.isnan(f[1])]))

        avrg_surface_distances_all.append((avrg_d_from_gt_to_pred, avrg_d_from_pred_to_gt))

    avrg_surface_distance_gt_to_pred = np.array([d[0] for d in avrg_surface_distances_all])
    avrg_surface_distance_gt_to_pred = [i for i in avrg_surface_distance_gt_to_pred if i < float(
        'inf')]  # remove inf from the list, which is for false negative objects

    avrg_surface_distance_pred_to_gt = np.array([d[1] for d in avrg_surface_distances_all])
    avrg_surface_distance_pred_to_gt = [i for i in avrg_surface_distance_pred_to_gt if
                                        not math.isnan(i)]  # remove NaNs from the list

    avrg_d_gt_to_pred_mean = np.mean(avrg_surface_distance_gt_to_pred)
    print(f"Average distance from the ground truth to predictions in microns: {avrg_d_gt_to_pred_mean * 1000}")
    avrg_d_from_pred_to_gt_mean = np.mean(avrg_surface_distance_pred_to_gt)
    print(f"Average distance from predictions to the ground truth in microns: {avrg_d_from_pred_to_gt_mean * 1000}")

    returned["Average distance from the ground truth to predictions in microns"] = avrg_d_gt_to_pred_mean * 1000
    returned["Average distance from predictions to the ground truth in microns"] = avrg_d_from_pred_to_gt_mean * 1000

    hausdorff_distances_all = []

    for surf_per_img in tqdm(surface_distances, desc="Computing robust hausdorff distances"):
        hausdorff_distances_all.append(
            np.mean(np.array([surface_distance.compute_robust_hausdorff(d, hausdorff_percent) for d in surf_per_img])))

    hausdorff_distances_all = [i for i in hausdorff_distances_all if i < float('inf')]  # remove inf from the list

    returned["Average Hausdorff distances"] = statistics.mean(hausdorff_distances_all)

    surf_overlap_all = []

    for surf_per_img in tqdm(surface_distances, desc="Computing surface overlap at tolerance"):
        surface_overlap = [surface_distance.compute_surface_overlap_at_tolerance(d, tolerance_mm=tolerance_mm) for d in
                           surf_per_img]

        surf_overlap_of_gt_surf_w_pred_surf = np.mean(np.array([f[0] for f in surface_overlap]))

        surf_overlap_of_pred_surf_w_gt_surf = np.mean(np.array([f[1] for f in surface_overlap]))

        surf_overlap_all.append((surf_overlap_of_gt_surf_w_pred_surf, surf_overlap_of_pred_surf_w_gt_surf))

    surf_overlap_of_gt_surf_w_pred_surf = np.array([d[0] for d in surf_overlap_all])
    surf_overlap_of_pred_surf_w_gt_surf = np.array(
        [d[1] for d in surf_overlap_all if not math.isnan(d[1])])  # remove NaNs from the list

    surf_overlap_of_gt_surf_w_pred_surf_mean = np.mean(surf_overlap_of_gt_surf_w_pred_surf)
    returned["Surface overlap fraction of the GT surfaces with the predicted surfaces"] = \
        surf_overlap_of_gt_surf_w_pred_surf_mean
    print(
        f"Surface overlap fraction of the GT surfaces with the predicted surfaces: "
        f"{surf_overlap_of_gt_surf_w_pred_surf_mean}")

    surf_overlap_of_pred_surf_w_gt_surf_mean = np.mean(surf_overlap_of_pred_surf_w_gt_surf)
    returned["Surface overlap fraction of the predicted surfaces with the GT surfaces"] = \
        surf_overlap_of_pred_surf_w_gt_surf_mean
    print(
        f"Surface overlap fraction of the predicted surfaces with the GT surfaces: "
        f"{surf_overlap_of_pred_surf_w_gt_surf_mean}")

    surf_dice_all = []

    for surf_per_img in tqdm(surface_distances, desc="Computing surface dice coefficients"):
        surface_dice = [surface_distance.compute_surface_dice_at_tolerance(d, tolerance_mm) for d in surf_per_img]

        surf_dice_all.append(statistics.mean(surface_dice))

    surf_dice_mean = statistics.mean(surf_dice_all)
    returned["Surface dice coefficients mean"] = surf_dice_mean
    print(f"Surface dice coefficients mean: {surf_dice_mean}")

    vol_dice_all = []

    for y_pred, y_true in tqdm(zip(predictions, ground_truth), total=len(predictions),
                               desc="Computing volumetric dice coefficients"):
        y_true, _, _ = relabel_sequential(y_true.astype("int32"))
        y_pred, _, _ = relabel_sequential(y_pred.astype("int32"))

        y_true, y_pred = group_matching_labels([y_true, y_pred], thresh=0.3)

        vol_dice_all.append(surface_distance.compute_dice_coefficient(y_true.astype("int32"), y_pred.astype("int32")))

    dice_coeffs_mean = statistics.mean(vol_dice_all)
    returned["Volume dice coefficients mean"] = dice_coeffs_mean
    print(f"Volume dice coefficients mean: {dice_coeffs_mean}")

    return returned


def brier_score(GT_labels, prediction):
    # convert from OCLarray to numpy array
    GT_arr = np.array(GT_labels)
    # change array type
    GT_arr = GT_arr.astype('int32')

    return brier_score_loss(GT_arr.flatten(), np.array(prediction).flatten())


def predict(image, clf_path, features):
    import apoc
    segmenter = apoc.ProbabilityMapper(opencl_filename=clf_path)
    labels = segmenter.predict(image, features=features)

    stats, _ = segmenter.statistics()
    df = pd.DataFrame(stats)
    swapped = df.swapaxes("index", "columns")
    swapped["sum"] = swapped.sum(axis=1)
    swapped

    return labels


def predict_pmaps_and_calculate_brier_scores(test_images, clf, features, GT_binary):

    predictions = []
    scores = []

    for i, img in enumerate(test_images):
        pred = predict(img, clf, features)
        predictions.append(pred)
        score = brier_score(GT_binary[i], pred)
        scores.append(score)

    return predictions, scores


def get_classifier_metrics_stats(path, clf_nr, fnames, GT_binary):

    mean_scores = []
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        predictions = []
        for fname in fnames:
            if clf_nr != 0:
                predictions.append(imread(
                    os.path.join(path, str(i), str(i), "results", "ProbabilityMapper" + str(clf_nr) + ".cl_results",
                                 fname)))
            else:
                predictions.append(
                    imread(os.path.join(path, str(i), str(i), "results", "ProbabilityMapper" + ".cl_results", fname)))
        scores_clf = []
        for gt, pred in zip(GT_binary, predictions):
            scores_clf.append(brier_score(gt, pred))
        mean_scores.append(mean(scores_clf))
    return mean_scores


def get_classifier_metrics_stats_log(path, clf_nr, fnames, GT_binary):
    mean_scores = []
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        predictions = []
        for fname in fnames:
            if clf_nr != 0:
                predictions.append(imread(
                    os.path.join(path, str(i), str(i), "results", "ProbabilityMapper" + str(clf_nr) + ".cl_results",
                                 fname)))
            else:
                predictions.append(
                    imread(os.path.join(path, str(i), str(i), "results", "ProbabilityMapper" + ".cl_results", fname)))
        scores_clf = []
        for gt, pred in zip(GT_binary, predictions):
            scores_clf.append(log_loss(gt.flatten(), pred.flatten()))
        mean_scores.append(mean(scores_clf))
    return mean_scores
