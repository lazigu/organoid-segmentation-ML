from __future__ import print_function, unicode_literals, absolute_import, division

import pickle
from pathlib import Path

import numpy as np
import logging
import os
from tifffile import imwrite, imread
from tqdm import tqdm

from src.config.config import combine_cfgs
from src.data.preprocess_utils import split_into_train_and_validation, load_files, resample_crop_downsample
from stardist import random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
import pyclesperanto_prototype as cle
from stardist.models import Config3D, StarDist3D
from src.data.preprocess_utils import load_dataset
from src.models.utils import get_available_gpus, augmenter

np.random.seed(42)
lbl_cmap = random_label_cmap()

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def stardist_predict(test_filepath, exp_dir, cfg=None):
    """
    From Stardist repository:
    Make sure to normalize the input image beforehand or supply a `normalizer` to the prediction function.

        Calling `model.predict_instances` will
        - predict object probabilities and star-convex polygon distances (see `model.predict` if you want those)
        - perform non-maximum suppression (with overlap threshold `nms_thresh`) for polygons above object probability
          threshold `prob_thresh`.
        - render all remaining polygon instances in a label image
        - return the label instances image and also the details (coordinates, etc.) of all remaining polygons

    """
    # load testing data and filenames
    _, fnames = load_files(test_filepath, step=1, return_fnames=True)

    # images = resample_crop_downsample(test_filepath, rescale=True, rescale_factor=[0.8, 0.8, 0.8])

    # load trained model
    model = StarDist3D(None, name='stardist', basedir=exp_dir)

    details_dir = os.path.join(exp_dir, "results", "details")
    Path(details_dir).mkdir(parents=True, exist_ok=True)
    probs_dir = os.path.join(exp_dir, "results", "probabilities")
    Path(probs_dir).mkdir(parents=True, exist_ok=True)
    distances_dir = os.path.join(exp_dir, "results", "distances")
    Path(distances_dir).mkdir(parents=True, exist_ok=True)

    # FACTOR = [0.5, 0.5, 0.5]

    for fname in tqdm(fnames, total=len(fnames)):

        img = imread(os.path.join(test_filepath, fname))
        print(f"Original shape: {img.shape}")

        if cfg is not None and cfg.DATASET.RESCALE:
            # downsample the image
            img = cle.resample(img,
                               # factor_x=FACTOR[2],
                               # factor_y=FACTOR[1],
                               # factor_z=FACTOR[0],
                               factor_x=cfg.DATASET.SCALE_FACTOR[2],
                               factor_y=cfg.DATASET.SCALE_FACTOR[1],
                               factor_z=cfg.DATASET.SCALE_FACTOR[0],
                               linear_interpolation=True)
            print(f"Scaled img shape: {img.shape}")

        (labels, details), (prob, distances) = model.predict_instances(img, verbose=True,
                                                                       # prob_thresh=cfg.MODEL.STARDIST.PROB_THRESHOLD,
                                                                       # nms_thresh=cfg.MODEL.STARDIST.NMS_THRESHOLD,
                                                                       # defaults prob_thresh=0.5 nms_thresh=0.4
                                                                       return_predict=True, n_tiles=(1, 1, 1))

        if cfg is not None and cfg.DATASET.RESCALE:
            # scale it back
            labels = cle.resample(labels,
                                  # factor_x=1 / FACTOR[2],
                                  # factor_y=1 / FACTOR[1],
                                  # factor_z=1 / FACTOR[0],
                                  factor_x=1 / cfg.DATASET.SCALE_FACTOR[2],
                                  factor_y=1 / cfg.DATASET.SCALE_FACTOR[1],
                                  factor_z=1 / cfg.DATASET.SCALE_FACTOR[0],
                                  linear_interpolation=False)

        # save labels
        imwrite(os.path.join(exp_dir, "results", fname + "_predicted_labels" + '.tif'), labels)

        # save details
        with open(os.path.join(details_dir, fname + '_details.pkl'), 'wb') as f:
            pickle.dump(details, f)

        # save probabilities
        imwrite(os.path.join(probs_dir, fname + '_probabilities.tif'), prob)

        # save star-convex polygon/polyhedra distances
        imwrite(os.path.join(distances_dir, fname + '_polyhedra_distances.tif'), distances)

        # import napari
        # viewer = napari.Viewer()
        #
        # viewer.add_image(img, name="original")
        # print(f"Details of {fname} labels: ")
        # print(details)
        # viewer.add_labels(labels, name="predicted_labels")
        # viewer.add_image(predict, name="stardist_prediction")
        # print(f"Details of {fname} prediction: ")
        # print(predict_details)

    # return labels, prob, distances


def stardist_train(raw_filepath, labels_filepath, cfg, output_path):
    """
    (1) calculation of the empirical anisotropy of labeled objects, (2) Config3D and StarDist3D
    model instances are created, (3) median size of objects in the training data is calculated and compared to the field
    of view of the network (it must be as big or smaller, otherwise, assertion error), (6) training of the model

    Parameters
    -----------
    raw_filepath : str
        directory containing images prepared for training
    labels_filepath : str
        directory containing labels prepared for training
    cfg : CfgNode
        combined configuration (defaults YACS + experiment YAML + .env)
    output_path : str
        directory where model and training logs are saved, if not specified - .experiments/expXX/stardist by default

    Returns
    -----------
    None

    """
    logger = logging.getLogger(__name__)

    # load prepared for training images and labels
    images, labels = load_dataset(rawdir=raw_filepath, labelsdir=labels_filepath, step=2)

    logger.info(
        f"{len(images)} images with shape of {images[0].shape} and {len(labels)} label images with shape "
        f"of {labels[0].shape} were loaded")

    X_train, Y_train, X_val, Y_val = split_into_train_and_validation(images, labels)
    labels=np.array(labels).astype("int32")
    extents = calculate_extents(labels)

    anisotropy = tuple(np.max(extents) / extents)
    logger.info(f"empirical anisotropy of labeled objects = {anisotropy}")

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = cfg.MODEL.USE_GPU and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    # cfg.MODEL.GRID = tuple([(int(1) if a > 1.5 else 2) for a in anisotropy])
    # print(len(cfg.MODEL.GRID))
    grid = (4, 4, 4)  # must be a tuple with items of power of 2
    cfg.MODEL.GRID = grid

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(cfg.MODEL.TRAIN.STARDIST_RAYS, anisotropy=anisotropy)

    config = Config3D(
        axes=cfg.DATASET.AXES,
        rays=rays,
        n_channel_in=cfg.DATASET.N_CHANNEL_IN,
        grid=cfg.MODEL.GRID,
        n_classes=cfg.DATASET.N_CLASSES,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        backbone=cfg.MODEL.BACKBONE.NAME,
        train_patch_size=cfg.MODEL.TRAIN.PATCHES_SIZE,
        train_batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
        train_epochs=cfg.MODEL.TRAIN.EPOCHS,
        train_steps_per_epoch=cfg.MODEL.TRAIN.STEPS_PER_EPOCH

    )
    logger.info(config)

    # tf.debugging.set_log_device_placement(True)
    get_available_gpus()
    # gpus = tf.config.list_logical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy(gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8, total_memory=8000)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    # with strategy.scope():
    model = StarDist3D(config, name='stardist', basedir=output_path)
    # model = StarDist3D(None, name='stardist', basedir=output_path) # uncomment if config is loaded from dir
    logger.info(f"StarDist3D has been built")

    median_size = calculate_extents(labels, np.median)
    fov = np.array(model._axes_tile_overlap('ZYX'))

    del images, labels

    logger.info(f"median object size:      {median_size}")
    logger.info(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")
        return

    print("Starting the training")

    history = model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=augmenter)

    import json
    # save history under the form of a json file
    json.dump(history.history, open(os.path.join(output_path, "history.json"), 'w'))

    # the optimized threshold values are saved to disk and will be automatically loaded with the model
    model.optimize_thresholds(X_val, Y_val)


def threshold_optimization(raw_filepath, labels_filepath, exp_dir, cfg_path):
    logger = logging.getLogger(__name__)

    logger.info(f'Getting experiment configuration file from {cfg_path}')
    cfg = combine_cfgs(cfg_path)

    # load prepared for training images and labels
    images, labels = load_dataset(rawdir=raw_filepath, labelsdir=labels_filepath)

    _, _, X_val, Y_val = split_into_train_and_validation(images, labels)

    del images
    del labels

    if cfg.DATASET.RESCALE:
        X_val = resample_crop_downsample(images=X_val, rescale=True, rescale_factor=cfg.DATASET.SCALE_FACTOR)
        Y_val = resample_crop_downsample(images=Y_val, rescale=True, rescale_factor=cfg.DATASET.SCALE_FACTOR)

    # load trained model
    model = StarDist3D(None, name='stardist', basedir=exp_dir)

    model.optimize_thresholds(X_val, np.array(Y_val).astype('int32')) #, predict_kwargs={"n_tiles": (4, 2, 2)})
