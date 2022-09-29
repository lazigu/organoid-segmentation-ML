import click
import logging
import os
from pathlib import Path
from datetime import date
import pyclesperanto_prototype as cle

from dotenv import find_dotenv, load_dotenv
from src.config.config import combine_cfgs
from src.data.preprocess_utils import preprocess, erode_labels, make_binary_labels, split_into_train_val_test, \
    split_into_timepoints, copy_data

# Load the .ENV path if it exists
if find_dotenv() != "":
    load_dotenv(find_dotenv(), verbose=True)

    PATH_DATA_RAW = Path(os.getenv("PATH_DATA_RAW"))
    PATH_LABELS = Path(os.getenv("PATH_LABELS"))


@click.command()
@click.argument('raw_filepath', envvar='PATH_DATA_RAW', type=click.Path(exists=True), required=False)
@click.argument('labels_filepath', envvar='PATH_LABELS', type=click.Path(exists=True), required=False)
@click.option('-cfg', '--cfg_path', default=None, type=click.Path(exists=True), required=False,
              help="Path to the YAML configuration file of the new experiment, which will be used "
                   "to overwrite default behaviour.")
def main(raw_filepath, labels_filepath, cfg_path):
    """
    Runs data pre-processing scripts to turn raw data and labels data into cleaned data ready to be used for
    segmentation algorithms. Saves all intermediate files in data/ directory

    Parameters
    ----------
    raw_filepath : str
        path to the directory with an original 4D raw file. There must be just one file in the directory, otherwise
        error will be thrown
    labels_filepath : str
        path to the directory with a 4D ground truth labels file. There must be just one file in the directory,
        otherwise error will be thrown
    cfg_path :
        path to the configuration YAML file
    """
    logger = logging.getLogger(__name__)

    logger.info(cfg_path)

    cle.select_device("RTX")
    # check which GPU is used
    logger.info(f"SELECTED DEVICE: {cle.get_device().name}")

    cfg = combine_cfgs(cfg_path)

    logger.info(f'Raw images path: {click.format_filename(raw_filepath)}')
    logger.info(f'Label images path: {click.format_filename(labels_filepath)}')

    base_dir = Path(__file__).resolve().parents[2]  # get directory of the project
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        logger.info(f"{data_dir} directory was created")

    original_dir = os.path.join(data_dir,
                                f"{cfg.DATASET.NAME}{'_' if len(cfg.DATASET.NAME) > 0 else ''}" + "original")

    # if original directory does not yet exist, create it and copy original data files
    if not os.path.exists(original_dir):
        os.mkdir(original_dir)
        logger.info(f"{original_dir} directory was created")

        # make a copy of original datasets to src/data/original/ directory if it does not exist yet
        # overwrite raw/labels_filepath variables to further process copies of original data
        raw_filepath = copy_data(raw_filepath, original_dir, "raw")
        labels_filepath = copy_data(labels_filepath, original_dir, "labels")
    else:
        logger.info(f"Existing directory found. Files for further processing will be loaded from there:"
                    f" \n {original_dir} ")

        # assert that there is more than one folder (raw and GT) in the existing directory
        assert len(os.listdir(original_dir)) > 1, f"Existing data/CFG.NAME_original directory \n {original_dir} " \
                                                  f"appears to be empty or contains only one folder inside. Please " \
                                                  f"delete this directory and run make_dataset again or change the " \
                                                  f"dataset name in CFG."

    # split 4D timelapse file into multiple 3D files
    if cfg.DATASET.SPLIT_TO_FRAMES:
        logger.info("Splitting into timepoints")
        split_into_timepoints(raw_filepath, save_dir=os.path.join(original_dir, "raw_timepoints"))
        split_into_timepoints(labels_filepath, save_dir=os.path.join(original_dir, "labels_timepoints"))

    raw_filepath = os.path.join(original_dir, "raw_timepoints")
    labels_filepath = os.path.join(original_dir, "labels_timepoints")

    # throw errors if there are no folders in data/original directory with multiple 3D images
    assert os.path.exists(os.path.join(original_dir, "labels_timepoints")), \
        "Please select ´DATASET: SPLIT_TO_FRAMES: True´ in make_dataset.YAML configuration file in order to get a " \
        "folder with multiple 3D files for further processing"

    assert os.path.exists(os.path.join(original_dir, "labels_timepoints")), \
        "Please select ´DATASET: SPLIT_TO_FRAMES: True´ in make_dataset.YAML configuration file in order to get a " \
        "folder with multiple 3D files for further processing"

    # todo ------------------------------------------------------------------------
    # todo: Vollseg takes .tiff files, for now all files are saved as .tif FIX THIS
    # todo ------------------------------------------------------------------------

    # define a path for processed files
    output_dir = os.path.join(base_dir, "data",
                              f"{cfg.DATASET.CFG_NAME}{'_' if len(cfg.DATASET.CFG_NAME) > 0 else ''}"
                              f"preprocessed_{date.today()}")

    # create a folder if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir_lbl = os.path.join(output_dir, "labels")
    # create a folder if it does not exist
    if not os.path.exists(output_dir_lbl):
        os.mkdir(output_dir_lbl)

    if cfg.DATASET.MAKE_ERODED_LABELS:
        erode_labels(labels_filepath, erosion_iterations=cfg.DATASET.EROSION_ITERATIONS,
                     save_dir=os.path.join(output_dir_lbl, "eroded"))
        # update labels path so the next step loads eroded label images
        labels_filepath = os.path.join(output_dir_lbl, "eroded")
        output_dir_lbl = os.path.join(output_dir_lbl, "eroded")

    if cfg.DATASET.MAKE_BINARY_LABELS:
        make_binary_labels(labels_filepath, save_path=os.path.join(output_dir_lbl, "binary"))
        labels_filepath = os.path.join(output_dir_lbl, "binary")

    # perform optional preprocessing steps like making images isotropic, cropping, scaling, normalizing and filling
    # small holes in label images
    raw_filepath, labels_filepath = preprocess(raw_filepath, labels_filepath, cfg, output_dir)

    # finally, split processed data into training, validation and test sets
    if cfg.DATASET.SPLIT_FOLDERS:
        split_into_train_val_test(raw_filepath, labels_filepath, cfg.DATASET.SPLIT_RATIO,
                                  os.path.join(base_dir, "data",
                                               f"{cfg.DATASET.CFG_NAME}{'_' if len(cfg.DATASET.CFG_NAME) > 0 else ''}"
                                               + f"train_val_test_{date.today()}"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
