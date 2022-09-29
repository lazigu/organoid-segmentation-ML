# todo allow selection on which gpu to run?
import os

from tifffile import imread
from vollseg import UNET, VollSeg, SmartSeeds3D
from tqdm import tqdm
from src.data.preprocess_utils import load_files
from src.tools.utils import load_json, write_yaml
from tensorflow.python.client import device_lib
import tensorflow as tf
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def write_prediction_cfg(cfg, exp_dir):
    """
    Combines training configuration from the model config.json file and prediction configuration that was defined in
    defaults config.py or experiment config files. Saves new configuration in the experiment folder as YAML file.

    Parameters
    ----------
    cfg : CfgNode
        Combined configuration in the priority order provided:
        .env > YAML exp config > default config
    exp_dir : str
        Path to the experiment´s folder.

    Returns
    ----------
    dict
        dictionary of full training/prediction configuration,
        which was also written to ..experiments/expXX/prediction_config.yaml

    """
    logger = logging.getLogger(__name__)

    # Opening JSON file with Vollseg model´s training configuration
    training_config = load_json(os.path.join(exp_dir, cfg.MODEL.PREDICTION.VOLLSEG.MODEL_NAME, "config.json"))

    config = {
        "training": {},
        "data": {
            "axes": cfg.DATASET.AXES,
            "rgb": cfg.DATASET.RGB,
        },
        "prediction": {
            "path": cfg.MODEL.PREDICTION.TEST_DIR,
            "model_dir": exp_dir,
            "model_name": cfg.MODEL.PREDICTION.VOLLSEG.MODEL_NAME,
            "save_dir": os.path.join(exp_dir, "results"),
            "n_tiles": cfg.MODEL.PREDICTION.VOLLSEG.N_TILES,
            "slice_merge": cfg.MODEL.PREDICTION.VOLLSEG.SLICE_MERGE,
            "unet_model": cfg.MODEL.PREDICTION.VOLLSEG.UNET_MODEL,
            "star_mode": cfg.MODEL.PREDICTION.VOLLSEG.STAR_MODEL,
            "roi_model": cfg.MODEL.PREDICTION.VOLLSEG.ROI_MODEL,
            "noise_model": cfg.MODEL.PREDICTION.VOLLSEG.NOISE_MODEL,
            "prob_thresh": cfg.MODEL.PREDICTION.VOLLSEG.PROB_THRESH,
            "nms_thresh": cfg.MODEL.PREDICTION.VOLLSEG.NMS_THRESH,
            "min_size_mask": cfg.MODEL.PREDICTION.VOLLSEG.MIN_SIZE_MASK,
            "min_size": cfg.MODEL.PREDICTION.VOLLSEG.MIN_SIZE,
            "max_size": cfg.MODEL.PREDICTION.VOLLSEG.MAX_SIZE,
            "use_probability": cfg.MODEL.PREDICTION.VOLLSEG.USE_PROBABILITY,
            "donormalize": cfg.MODEL.PREDICTION.VOLLSEG.DONORMALIZE,
            "lower_perc": cfg.MODEL.PREDICTION.VOLLSEG.LOWER_PERC,
            "upper_perc": cfg.MODEL.PREDICTION.VOLLSEG.UPPER_PERC,
            "dounet": cfg.MODEL.PREDICTION.VOLLSEG.DOUNET,
            "seedpool": cfg.MODEL.PREDICTION.VOLLSEG.SEEDPOOL,
            "iou_threshold": cfg.MODEL.PREDICTION.VOLLSEG.IOU_THRESHOLD,
        }
    }

    config["training"].update(training_config)
    write_yaml(config, os.path.join(exp_dir, "prediction_config.yaml"))
    logger.info(f"Prediction configuration was saved to: {os.path.join(exp_dir, 'prediction_config.yaml')}")


def vollseg_predict(test_filepath, cfg, exp_dir):
    """Performs Vollseg prediction and saves predicted labels to ..experiments/expXX/results directory

    Parameters
    -----------
    test_filepath: str
        path to the folder with test images
    cfg: CfgNode
        combined configuration in the priority order provided:
        .env > YAML exp config > default config
    exp_dir: str
        path to experiment´s folder, e.g. src/experiments/expXX

    Returns
    -----------
    None

    """

    logger = logging.getLogger(__name__)

    _, fnames = load_files(test_filepath, return_fnames=True)
    logger.info(f"Prediction running on {len(fnames)} test images...")

    # training_config = Config(**(load_json(os.path.join(exp_dir, cfg.MODEL.PREDICTION.MODEL_NAME, "config.json"))))

    write_prediction_cfg(cfg, exp_dir)

    for fname in tqdm(fnames):

        image = imread(os.path.join(test_filepath, fname))

        # config is automatically loaded from model´s directory
        model = UNET(config=None, name=cfg.MODEL.PREDICTION.VOLLSEG.MODEL_NAME, basedir=exp_dir)

        VollSeg(image,
                unet_model=model,
                # star_model=StarDist3D(config=None, name=cfg.MODEL.PREDICTION.MODEL_NAME, basedir=exp_dir),
                # todo: take care when multiple models are used
                # roi_model=MASKUNET(config=None, name=cfg.MODEL.PREDICTION.MODEL_NAME, basedir=exp_dir),  # todo
                axes=cfg.DATASET.AXES,
                # noise_model=CARE(config=None, name=cfg.MODEL.PREDICTION.MODEL_NAME, basedir=exp_dir),  # todo
                prob_thresh=cfg.MODEL.PREDICTION.VOLLSEG.PROB_THRESH,
                nms_thresh=cfg.MODEL.PREDICTION.VOLLSEG.NMS_THRESH,
                min_size_mask=cfg.MODEL.PREDICTION.VOLLSEG.MIN_SIZE_MASK,
                min_size=cfg.MODEL.PREDICTION.VOLLSEG.MIN_SIZE,
                max_size=cfg.MODEL.PREDICTION.VOLLSEG.MAX_SIZE,
                n_tiles=cfg.MODEL.PREDICTION.VOLLSEG.N_TILES,
                UseProbability=cfg.MODEL.PREDICTION.VOLLSEG.USE_PROBABILITY,
                donormalize=cfg.MODEL.PREDICTION.VOLLSEG.DONORMALIZE,
                lower_perc=cfg.MODEL.PREDICTION.VOLLSEG.LOWER_PERC,
                upper_perc=cfg.MODEL.PREDICTION.VOLLSEG.UPPER_PERC,
                dounet=cfg.MODEL.PREDICTION.VOLLSEG.DOUNET,
                seedpool=cfg.MODEL.PREDICTION.VOLLSEG.SEEDPOOL,
                save_dir=os.path.join(exp_dir, "results/"),
                Name=fname.split(".")[0],
                slice_merge=cfg.MODEL.PREDICTION.VOLLSEG.SLICE_MERGE,
                iou_threshold=cfg.MODEL.PREDICTION.VOLLSEG.IOU_THRESHOLD,
                RGB=cfg.DATASET.RGB
                )


def vollseg_train(images_filepath, labels_filepath, cfg, exp_dir):
    """
    Performs Vollseg training of the selected models (specified in the experiment config YAML file) and saves the
    trained model, training config and tensorboard logs to  ..experiments/expXX directory

    Parameters
    -----------
    images_filepath: str
        path to the folder with training images
    labels_filepath: str
        path to the folder with labels images
    cfg: CfgNode
        combined configuration in the priority order provided:
        .env > YAML exp config > default config
    exp_dir: str
        path to experiment´s folder, e.g. src/experiments/expXX

    """

    get_available_gpus()

    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')

    # if to train on multiple gpus simultaneously
    strategy = tf.distribute.MirroredStrategy(gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        SmartSeeds3D(base_dir="",
                     npz_filename=str(os.path.split(exp_dir)[1].split('.')[0]) + "_train_data",
                     model_name=cfg.MODEL.PREDICTION.VOLLSEG.MODEL_NAME,
                     model_dir=exp_dir,
                     raw_dir=images_filepath,
                     #real_mask_dir=labels_filepath,
                     binary_mask_dir=labels_filepath,
                     n_channel_in=cfg.DATASET.N_CHANNEL_IN,
                     backbone=cfg.MODEL.BACKBONE.NAME,
                     load_data_sequence=True,
                     validation_split=cfg.MODEL.TRAIN.VOLLSEG.VALIDATION_SPLIT,
                     n_patches_per_image=cfg.MODEL.TRAIN.VOLLSEG.N_PATCHES_PER_IMAGE,
                     generate_npz=cfg.MODEL.TRAIN.VOLLSEG.GENERATE_NPZ,
                     train_unet=cfg.MODEL.TRAIN.VOLLSEG.TRAIN_UNET,
                     use_gpu=cfg.MODEL.USE_GPU,
                     patch_x=cfg.MODEL.TRAIN.PATCHES_SIZE[2],
                     patch_y=cfg.MODEL.TRAIN.PATCHES_SIZE[1],
                     patch_z=cfg.MODEL.TRAIN.PATCHES_SIZE[0],
                     batch_size=cfg.MODEL.TRAIN.BATCH_SIZE,
                     epochs=cfg.MODEL.TRAIN.EPOCHS,
                     learning_rate=cfg.MODEL.TRAIN.LEARNING_RATE,
                     # UNET parameters:
                     depth=cfg.MODEL.BACKBONE.UNET.N_DEPTH,
                     kern_size=cfg.MODEL.BACKBONE.UNET.KERNEL_SIZE,
                     startfilter=cfg.MODEL.BACKBONE.UNET.N_FILTER_BASE,
                     )
