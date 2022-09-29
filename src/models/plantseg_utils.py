import logging
from src.tools.utils import load_yaml, write_yaml


STRIDE_MENU = {
    "accurate": 0.5,
    "balanced": 0.75,
    "draft": 0.9
}


def get_stride_shape(patch_shape, stride_key):
    return [max(int(p * STRIDE_MENU[stride_key]), 1) for p in patch_shape]


def convert_to_plantseg_config(config, path):
    logger = logging.getLogger(__name__)
    logger.info('Converting experiment yaml file to plantseg yaml config file...')

    CFG = config.MODEL.PLANTSEG

    for p in config.DATASET.PATCHES_SIZE:
        print(p)

    config = {
        "path": config.MODEL.PREDICTION.TEST_DIR,
        "preprocessing": {
            "state": CFG.PREPROCESSING.ACTIVE,
            "save_directory": CFG.PREPROCESSING.SAVE_DIRECTORY,
            "factor": CFG.PREPROCESSING.FACTOR,
            "order": CFG.PREPROCESSING.ORDER,
            "crop_volume": CFG.PREPROCESSING.CROP_VOLUME,
            "filter": {
                "state": CFG.PREPROCESSING.FILTER.STATE,
                "type": CFG.PREPROCESSING.FILTER.TYPE,
                "filter_param": CFG.PREPROCESSING.FILTER.PARAM,
            }
        },

        "cnn_prediction": {
            "state": CFG.CNN.PREDICTION.STATE,
            "model_name": CFG.CNN.PREDICTION.MODEL_NAME,
            "device": CFG.CNN.PREDICTION.DEVICE,
            "mirror_padding": CFG.CNN.PREDICTION.MIRROR_PADDING,
            "patch_halo": CFG.CNN.PREDICTION.PATCH_HALO,
            "num_workers": CFG.CNN.PREDICTION.NUM_WORKERS,
            "patch": config.DATASET.PATCHES_SIZE,
            "stride": [max(int(p * STRIDE_MENU[CFG.CNN.PREDICTION.STRIDE]), 1) for p in
                       config.DATASET.PATCHES_SIZE],
            "version": CFG.CNN.PREDICTION.VERSION,
            "model_update": CFG.CNN.PREDICTION.MODEL_UPDATE,
        },

        "cnn_postprocessing": {
            "state": CFG.CNN.POSTPROCESSING.STATE,
            "tiff": CFG.CNN.POSTPROCESSING.TIFF,
            "output_type": CFG.CNN.POSTPROCESSING.OUTPUT_TYPE,
            "factor": CFG.CNN.POSTPROCESSING.RESCALING_FACTOR,
            "order": CFG.CNN.POSTPROCESSING.SPLINE_ORDER,
            "save_raw": CFG.CNN.POSTPROCESSING.SAVE_RAW,
        },

        "segmentation":  {
            "state": CFG.SEGMENTATION.STATE,
            "name": CFG.SEGMENTATION.NAME,
            "beta": CFG.SEGMENTATION.BETA,
            "save_directory": CFG.SEGMENTATION.SAVE_DIRECTORY,
            "run_ws": CFG.SEGMENTATION.RUN_WS,
            "ws_2D": CFG.SEGMENTATION.WS_2D,
            "ws_threshold": CFG.SEGMENTATION.WS_THRESHOLD,
            "ws_minsize": CFG.SEGMENTATION.WS_MINSIZE,
            "ws_sigma": CFG.SEGMENTATION.WS_SIGMA,
            "ws_w_sigma": CFG.SEGMENTATION.WS_W_SIGMA,
            "post_minsize": CFG.SEGMENTATION.POST_MINSIZE,
        },

        "segmentation_postprocessing": {
            "state": CFG.SEGMENTATION.POSTPROCESSING.STATE,
            "tiff": CFG.SEGMENTATION.POSTPROCESSING.TIFF,
            "factor": CFG.SEGMENTATION.POSTPROCESSING.RESCALING_FACTOR,
            "order": CFG.SEGMENTATION.POSTPROCESSING.SPLINE_ORDER,
            "save_raw": CFG.SEGMENTATION.POSTPROCESSING.SAVE_RAW
        }
    }

    write_yaml(config, path)
    print(f"Plantseg config file has been successfully written to: {path}")
    return load_yaml(path)

