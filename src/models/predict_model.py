from datetime import date

import click
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from src.config.config import combine_cfgs

# Load the .ENV path.
from src.tools.utils import write_yaml

if find_dotenv() != "":
    load_dotenv(find_dotenv(), verbose=True)
    if os.getenv("PATH_DATA_TEST") is not None:
        PATH_DATA_TEST = Path(os.getenv("PATH_DATA_TEST"))


@click.command()
@click.argument('test_filepath', envvar='PATH_DATA_TEST', type=click.Path(exists=True), required=False)
@click.option('-cfg', '--cfg_path', default=None, type=click.Path(exists=True), required=False,
              help="Path to the YAML configuration file of the new experiment, which will be used "
                   "to overwrite default behaviour.")
def main(test_filepath, cfg_path):
    """
    Runs prediction scripts to predict segmentation for test data when path
    is provided in the .env file or given as an argument in CLI

    """
    logger = logging.getLogger(__name__)
    logger.info(f'Getting experiment configuration file from {cfg_path}')

    exp = os.path.split(cfg_path)[1].split('.')[0]  # get experiment name e.g. "exp06"

    cfg = combine_cfgs(cfg_path)
    # print(cfg)
    base_dir = Path(__file__).resolve().parents[2]  # get directory of the project

    exp_dir = os.path.join(base_dir, "experiments", exp)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    results_dir = os.path.join(exp_dir, "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # save yaml file with paths that the command was run in the terminal, just in case:
    data = {
        "TEST_PATH": test_filepath,
        "CFG_PATH": cfg_path
    }
    write_yaml(data, os.path.join(exp_dir, "paths_predict_" + str(date.today()) + ".yaml"))

    if cfg.MODEL.CLASSIFIER_HEAD.lower() == "plantseg":
        from plantseg.pipeline.raw2seg import raw2seg
        from src.models.plantseg_utils import convert_to_plantseg_config
        logger.info("Plantseg prediction in progress")

        preprocessing_dir = os.path.join(results_dir, cfg.MODEL.PLANTSEG.PREPROCESSING.SAVE_DIRECTORY)
        cfg.MODEL.PLANTSEG.PREPROCESSING.SAVE_DIRECTORY = preprocessing_dir
        # make preprocessing results folders if it does not exist
        if not os.path.exists(preprocessing_dir):
            os.mkdir(preprocessing_dir)

        # make segmentation results folders if it does not exist
        segmentation_dir = os.path.join(results_dir, cfg.MODEL.PLANTSEG.SEGMENTATION.SAVE_DIRECTORY)
        cfg.MODEL.PLANTSEG.SEGMENTATION.SAVE_DIRECTORY = segmentation_dir
        if not os.path.exists(segmentation_dir):
            os.mkdir(segmentation_dir)

        plantseg_cfg = convert_to_plantseg_config(cfg, os.path.join(exp_dir, "plantseg_cfg.yaml"))

        # run Plantseg pipeline
        raw2seg(plantseg_cfg)

    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "vollseg":
        from src.models.vollseg import vollseg_predict
        logger.info("Vollseg prediction in progress")
        vollseg_predict(test_filepath, cfg, exp_dir)

    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "apoc":
        from src.models.apoc import apoc_predict
        logger.info("APOC classifier prediction in progress")
        apoc_predict(filespath=test_filepath, exp_dir=exp_dir, cfg=cfg)

    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "stardist":
        from src.models.stardist3D import stardist_predict
        logger.info("Stardist prediction in progress")
        stardist_predict(test_filepath, exp_dir, cfg)

    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "unet":
        from src.models.unet import unet_predict
        logger.info("UNET semantic segmentation prediction in progress")
        unet_predict(test_filepath, cfg, exp_dir)

    else:
        logger.info("No classifier head selected in config")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
