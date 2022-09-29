from datetime import date

import click
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from src.config.config import combine_cfgs

# Load the .ENV path.
from src.models.unet import unet_train
from src.models.utils import copy_model_to_backup
from src.tools.utils import write_yaml

if find_dotenv() != "":
    load_dotenv(find_dotenv(), verbose=True)
    if os.getenv("PATH_DATA_RAW") is not None:
        PATH_DATA_RAW = Path(os.getenv("PATH_DATA_RAW"))   # if PATH_DATA_RAW is provided in .env
    if os.getenv("PATH_LABELS") is not None:
        PATH_LABELS = Path(os.getenv("PATH_LABELS"))       # if PATH_LABELS is provided in .env


@click.command()
@click.argument('raw_filepath', envvar='PATH_DATA_RAW', type=click.Path(exists=True), required=False)
@click.argument('labels_filepath', envvar='PATH_LABELS', type=click.Path(exists=True), required=False)
@click.option('-cfg', '--cfg_path', default=None, type=click.Path(exists=True), required=False,
              help="Path to the YAML configuration file of the new experiment, which will be used "
                   "to overwrite default behaviour.")
def main(raw_filepath, labels_filepath, cfg_path):
    """
    Runs training scripts to train a segmentation model
    """
    logger = logging.getLogger(__name__)

    logger.info(f'Getting experiment configuration file from {cfg_path}')
    cfg = combine_cfgs(cfg_path)

    base_dir = Path(__file__).resolve().parents[2]   # get directory of the project
    exp = os.path.split(cfg_path)[1].split('.')[0]   # get experiment name e.g. "exp06"

    exp_dir = os.path.join(base_dir, "experiments", exp)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        logger.info(f"Output directory was created: {exp_dir}")

    # create output dir if it does not exist
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        logger.info(f"Output directory was created: {exp_dir}")

    # save yaml file with paths that the command was run in the terminal, just in case:
    data = {
        "IMAGES_PATH": raw_filepath,
        "LABELS_PATH": labels_filepath,
        "CFG_PATH": cfg_path
    }
    write_yaml(data, os.path.join(exp_dir, "paths_train_" + str(date.today()) + ".yaml"))

    # make result folders if they do not exist
    results_dir = os.path.join(exp_dir, "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        logger.info(f"Results directory was created: {results_dir}")

    if cfg.MODEL.CLASSIFIER_HEAD.lower() == "stardist":
        from src.models.stardist3D import stardist_train
        stardist_train(raw_filepath, labels_filepath, cfg, exp_dir)

        copy_model_to_backup(os.path.join(exp_dir, "stardist"), exp_dir)

    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "vollseg":
        from src.models.vollseg import vollseg_train
        vollseg_train(raw_filepath, labels_filepath, cfg, exp_dir)

    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "apoc":
        from src.models.apoc import apoc_train
        apoc_train(exp_dir, raw_filepath, labels_filepath, cfg)
    elif cfg.MODEL.CLASSIFIER_HEAD.lower() == "unet":
        unet_train(raw_filepath, labels_filepath, exp_dir, cfg=cfg)
    else:
        logger.info("No CLASSIFIER_HEAD selected in experiment´s configuration YAML file")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

