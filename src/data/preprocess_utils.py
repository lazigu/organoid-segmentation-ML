import logging
import os
import shutil
import tempfile
from glob import glob
from pathlib import Path

from shutil import copyfile
from skimage.io import imread
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import pyclesperanto_prototype as cle
from scipy import ndimage
from tifffile import imwrite
from typing import List

from src.config.config import get_cfg_defaults


def preprocess(raw_dirpath, labels_dirpath=None, cfg=None, output_dir=None):
    """
    Utility function combining preprocessing functions resample_crop_downsample, normalize_data, and fill_small_holes
    with a given parameters in the configuration file, and appropriate folder creation for preprocessed data.

    Parameters
    ----------
    raw_dirpath : str
        path to a folder containing multiple 3D images
    labels_dirpath : str
        path to a folder containing multiple 3D label images
    cfg : CfgNode
        combined configuration
    output_dir : str
        directory ./data/name_processed

    """

    logger = logging.getLogger(__name__)

    if cfg is None:
        cfg = get_cfg_defaults()
    if output_dir is None:
        output_dir = os.path.join(Path(__file__).resolve().parents[2], "data", "preprocessed")

    out_raw_dirpath = os.path.join(output_dir, "images")
    # create a folder if it does not exist
    if not os.path.exists(out_raw_dirpath):
        Path(out_raw_dirpath).mkdir(parents=True, exist_ok=True)

    out_labels_dirpath = os.path.join(output_dir, "labels")
    if labels_dirpath is not None:
        # create a folder if it does not exist
        if not os.path.exists(out_labels_dirpath):
            Path(out_labels_dirpath).mkdir(parents=True, exist_ok=True)

    if cfg.DATASET.MAKE_ISOTROPIC:
        path_img_isotropic = os.path.join(out_raw_dirpath, "isotropic")

        voxel_size = np.array(cfg.DATASET.VOXEL_SIZE)
        norm_voxel_size = voxel_size / voxel_size[2]

        logger.info(f"Raw data is being rescaled with a factor determined from the the given voxel size:"
                    f" {norm_voxel_size}...")
        # resample image data to make it isotropic and crop if chosen, and save results
        resample_crop_downsample(raw_dirpath, make_isotropic=True,
                                 voxel_size=cfg.DATASET.VOXEL_SIZE,
                                 linear_interpolation=True, save_dir=path_img_isotropic)
        raw_dirpath = path_img_isotropic

        if labels_dirpath is not None:
            path_lbl_isotropic = os.path.join(out_labels_dirpath, "isotropic")
            logger.info(f"Labels data is being rescaled with a factor determined from the the given voxel size:"
                        f" {norm_voxel_size}...")
            resample_crop_downsample(labels_dirpath, make_isotropic=True,
                                     voxel_size=cfg.DATASET.VOXEL_SIZE,
                                     linear_interpolation=False, save_dir=path_lbl_isotropic)
            # overwrite path variables
            labels_dirpath = path_lbl_isotropic

    if cfg.DATASET.CROP:
        path_img_cropped = os.path.join(raw_dirpath, "cropped")

        logger.info(f"Cropping raw images...")
        # crop images, and save results
        resample_crop_downsample(raw_dirpath, crop=cfg.DATASET.CROP, size=cfg.DATASET.CROP_SIZE,
                                 save_dir=path_img_cropped)
        # overwrite path variables
        raw_dirpath = path_img_cropped

        if labels_dirpath is not None:
            path_lbl_cropped = os.path.join(labels_dirpath, "cropped")
            logger.info(f"Cropping labels images...")
            resample_crop_downsample(labels_dirpath, crop=cfg.DATASET.CROP, size=cfg.DATASET.CROP_SIZE,
                                     save_dir=path_lbl_cropped)
            labels_dirpath = path_lbl_cropped

    if cfg.DATASET.RESCALE:
        path_img_scaled = os.path.join(raw_dirpath, "scaled")

        logger.info(f"Raw data is being rescaled with a factor of {cfg.DATASET.SCALE_FACTOR}...")
        resample_crop_downsample(raw_dirpath, rescale=True, rescale_factor=cfg.DATASET.SCALE_FACTOR,
                                 linear_interpolation=True, save_dir=path_img_scaled)
        # overwrite path variables
        raw_dirpath = path_img_scaled

        if labels_dirpath is not None:
            path_lbl_scaled = os.path.join(labels_dirpath, "scaled")
            logger.info(f"Labels data is being rescaled with a factor of {cfg.DATASET.SCALE_FACTOR}...")
            resample_crop_downsample(labels_dirpath, rescale=True, rescale_factor=cfg.DATASET.SCALE_FACTOR,
                                     linear_interpolation=False, save_dir=path_lbl_scaled)
            labels_dirpath = path_lbl_scaled

    if cfg.DATASET.NORMALIZE:

        path_normalized = os.path.join(raw_dirpath, "normalized")

        images = load_files(raw_dirpath)

        if not os.path.exists(path_normalized) or len(path_normalized) == 0:
            normalize_data(images, n_channel=cfg.DATASET.N_CHANNEL_IN, save_dir=path_normalized)

        raw_dirpath = path_normalized

    if cfg.DATASET.FILL_HOLES and labels_dirpath is not None:

        path_filled_holes = os.path.join(labels_dirpath, "filled_holes")

        labels = load_files(labels_dirpath)

        if not os.path.exists(path_filled_holes) or len(path_filled_holes) == 0:
            fill_small_holes(labels, save_dir=path_filled_holes)

        # update labels path variable
        labels_dirpath = path_filled_holes

    if labels_dirpath is not None:
        return raw_dirpath, labels_dirpath
    else:
        return raw_dirpath


def load_dataset(rawdir: str = None, labelsdir: str = None, return_fnames: bool = False,
                 return_only_fnames: bool = False, step: int = 1):
    """
    Parameters
    -------------
    rawdir : str
        directory of images
    labelsdir : str
        directory of labels
    return_fnames : bool
        whether to return both filenames and loaded data
    return_only_fnames : bool
        whether to return only filenames in given directories and not loaded data
    step : int

    Returns
    -------------
    List of lists of images, labels or filenames or list of two tuples [(image, filenames), (images, filenames)]
        A list of lists (images and labels or image and label filenames) or a list of tuples if chosen to return both
        filenames and loaded data
    """

    returned = []
    if rawdir is not None:
        if return_only_fnames:
            fnames = load_files(rawdir, step=step, return_fnames=return_fnames,
                                return_only_fnames=return_only_fnames)
            returned.append(fnames)
        elif return_fnames and not return_only_fnames:
            images, fnames = load_files(rawdir, step=step, return_fnames=return_fnames)
            returned.append((images, fnames))
        else:
            images = load_files(rawdir, step=step)
            returned.append(images)

    if labelsdir is not None:
        if return_only_fnames:
            fnames = load_files(labelsdir, step=step, return_fnames=return_fnames,
                                return_only_fnames=return_only_fnames)
            returned.append(fnames)
        elif return_fnames and not return_only_fnames:
            labels, fnames = load_files(labelsdir, step=step, return_fnames=return_fnames)
            returned.append((labels, fnames))
        else:
            labels = load_files(labelsdir, step=step)
            returned.append(labels)

    return returned


def load_files(files_dir, step=1, start=0, stop=None, return_fnames=False, return_only_fnames=False):
    """
    Utility function to load images

    Parameters
    ------------
    files_dir : str
        directory of raw images
    start : int
    stop : int
    step : int
    return_fnames : bool
        whether to return also original loaded files names
    return_only_fnames : bool
        whether to return only filenames

    Returns
    ------------
    A list of images or a tuple(images, filenames)

    """

    logger = logging.getLogger(__name__)
    print(f"Loading files with step size: {step} from {files_dir}")

    filenames = natsorted(glob(os.path.join(files_dir, "*.tif")))

    if len(filenames) == 0:
        filenames = natsorted(glob(os.path.join(files_dir, "*.tiff")))

    if return_only_fnames:
        fnames = [os.path.split(file)[1] for file in filenames[::step]]
        return fnames
    elif not return_only_fnames and return_fnames:
        images = list(map(imread, filenames[start:: step]))
        fnames = [os.path.split(file)[1] for file in filenames[::step]]
        assert len(images) > 0, f"No files in the directory match pattern *.tif. Please change the directory."
        logger.info(f"Loaded {len(images)} data files from {files_dir} with shape {images[0].shape}")
        return images, fnames
    else:
        images = list(map(imread, filenames[start: stop: step]))
        logger.info(f"Loaded {len(images)} data files from {files_dir} with shape {images[0].shape}")
        return images


def resample_crop_downsample(files_path=None, images=None, fnames=None, voxel_size: List = None,
                             make_isotropic: bool = False, crop: bool = False, size: List = None, rescale: bool = False,
                             rescale_factor: List = None, linear_interpolation: bool = False,
                             save_dir: str = None) -> List:
    """
    Resamples, crops, scales up/down the given list of images or images from a given directory.

    Parameters
    -----------
    files_path : str
            path to a folder containing images to be resampled/cropped
    voxel_size : list
            a list of voxel sizes to make the image isotropic, default [1, 1, 1]
    make_isotropic : bool
            perform resampling, default False
    crop : bool
            perform cropping, default False
    size : list
            cropping parameters - image[zx:zy, h:w, y:x]
    rescale : bool
            scale the image up/down, default False
    rescale_factor : list
            size factors for rescaling, default [1, 1, 1]
    linear_interpolation : bool
            perform linear interpolation during resampling, default False
    save_dir : str
            directory where processed images will be saved
    fnames : list
            a list of filenames that processed images should be named when saved, by default 1.tif, 2.tif, 3.tif...
            must be same length as a list of original images

    Returns
    -----------
    images : list
        a list of resampled/cropped images

    """
    logger = logging.getLogger(__name__)

    assert files_path is not None or len(images) > 0, "Files path or loaded images not provided"

    if files_path is not None:
        fnames = load_files(files_path, return_fnames=True, return_only_fnames=True)

    if fnames is None:
        fnames = [str(i) for i in enumerate(images)]

    processed_images = []

    # assert len(images) > 0
    # if fnames is not None:
    #     assert len(fnames) == len(images), "The length of filenames list must be same as the length of images"

    # assert len(images[0].shape) == 3  # make sure we are dealing with 3D images

    # logger.info(f"Original image shape: {images[0].shape}")

    for name in tqdm(fnames, total=len(fnames), desc="Preprocessing files"):

        image = imread(os.path.join(files_path, name))

        image_temp = cle.push(np.copy(image))

        # voxel size determined from the metadata and normalisation is performed
        if make_isotropic:
            voxel_size = np.array(voxel_size)
            norm_voxel_size = voxel_size / voxel_size[2]

            image_temp = cle.resample(image_temp, factor_x=norm_voxel_size[2], factor_y=norm_voxel_size[1],
                                      factor_z=norm_voxel_size[0], linear_interpolation=linear_interpolation)
            # image_temp = np.array(image_temp)[size[0]:size[1], size[2]:size[3], size[4]:size[5]]

        if crop:
            image_temp = np.array(image_temp)[size[0]:size[1], size[2]:size[3], size[4]:size[5]]

        if rescale:
            image_temp = cle.pull(cle.resample(image_temp, factor_x=rescale_factor[2], factor_y=rescale_factor[1],
                                               factor_z=rescale_factor[0], linear_interpolation=linear_interpolation))

        if save_dir is not None:
            if not os.path.exists(save_dir):
                Path(save_dir).mkdir(parents=True, exist_ok=True)

            imwrite(os.path.join(save_dir, name), image_temp)

        processed_images.append(cle.pull(image_temp))

    logger.info(f"Image shape after processing: {processed_images[0].shape}")
    logger.info(f"Processed images saved to: {save_dir}")

    return processed_images


def _erode_mask(segmentation_labels, label_id, erosion_iterations, structuring_element=None):
    # pixel which belongs to the current mask is equal 1, all others - 0
    only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)

    eroded = ndimage.binary_erosion(input=only_current_label_id, structure=structuring_element,
                                    iterations=erosion_iterations).astype(
        segmentation_labels.dtype)

    relabeled_eroded = np.where(eroded == 1, label_id, 0)

    return relabeled_eroded


def _erode_labels(segmentation, erosion_iterations, structuring_element=None):
    # create empty list where the eroded masks can be saved to
    list_of_eroded_masks = []

    # iterate through each mask in the image
    for i in range(1, segmentation.max() + 1):
        label_id = i
        list_of_eroded_masks.append(_erode_mask(segmentation, label_id, erosion_iterations, structuring_element))

    # convert list of numpy arrays to stacked numpy array
    final_array = np.stack(list_of_eroded_masks)

    # max_IP to reduce the stack of arrays, each containing one labelled region, to a single 2D np array.
    final_array_labelled = np.sum(final_array, axis=0)

    return final_array_labelled


def erode_labels(lbl_path=None, labels=None, erosion_iterations=1, save: bool = False, save_dir=None,
                 structuring_element=None, return_final_eroded_list=False) -> List:
    """
    Erodes given label images by a specified number of pixels (erosion_iterations).
    See more: https://forum.image.sc/t/shrink-labeled-regions/50443/10

    Parameters
    -----------
    lbl_path : str
        directory of a folder with label images
    labels : list
        a list of label images already loaded to disk
    erosion_iterations : int
        how many pixels erode from edges, int default = 1
    save : bool
        if save processed files
    save_dir : str
        directory where eroded label images are saved, default = labels_path + "/eroded"

    Returns
    -----------
    final_eroded_list : list
        a list of eroded labels images

    """
    logger = logging.getLogger(__name__)

    if lbl_path is not None:
        logger.info(f"Eroding labels from: {lbl_path}")
        _, fnames = load_files(files_dir=lbl_path, return_fnames=True)
    else:
        fnames = [str(i) + ".tif" for i in enumerate(labels)]

    if return_final_eroded_list:
        final_eroded_list = []

    for i, name in tqdm(enumerate(fnames), desc="Eroding labels"):
        # mask = lbl.astype('int32')
        if lbl_path is not None:
            mask = imread(os.path.join(lbl_path, name))
        else:
            mask = labels[i]
        eroded_label = []

        for z_slice in range(mask.shape[0]):
            # if there are no labels at all in the 2D image, do not perform any erosion but add it to a new list
            if mask[z_slice].max() == 0:  # add or mask[z_slice].max() == 1 if to not erode where there is just one label
                eroded_label.append(mask[z_slice])
                continue

            # erode all masks in one z-slice by one pixel
            # add eroded image to a new list
            eroded = _erode_labels(mask[z_slice], erosion_iterations, structuring_element)
            eroded_label.append(eroded)

        # convert list of numpy arrays to stacked numpy array to get a 3D image again with eroded masks
        final_eroded = np.stack(eroded_label)

        if return_final_eroded_list:
            final_eroded_list.append(final_eroded)

        if save:
            if save_dir is None:
                save_dir = os.path.join(lbl_path, "eroded")

            if not os.path.exists(save_dir):
                Path(save_dir).mkdir(parents=True, exist_ok=True)

            imwrite(os.path.join(save_dir, name), final_eroded)
    if save_dir is not None:
        logger.info(f"Eroded labels saved to: {save_dir}")

    if return_final_eroded_list:
        return final_eroded_list


def _dilate_mask(segmentation_labels, label_id, dilation_iterations, structuring_element=None):
    # pixel which belongs to the current mask is equal 1, all others - 0
    only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)

    eroded = ndimage.binary_dilation(input=only_current_label_id, structure=structuring_element,
                                     iterations=dilation_iterations).astype(
        segmentation_labels.dtype)

    relabeled_eroded = np.where(eroded == 1, label_id, 0)

    return relabeled_eroded


def _dilate_labels(segmentation, dilation_iterations, structuring_element=None):
    # create empty list where the eroded masks can be saved to
    list_of_dilated_masks = []

    # iterate through each mask in the image
    for i in range(1, segmentation.max() + 1):
        label_id = i
        list_of_dilated_masks.append(_dilate_mask(segmentation, label_id, dilation_iterations,
                                                  structuring_element=structuring_element))

    # convert list of numpy arrays to stacked numpy array
    final_array = np.stack(list_of_dilated_masks)

    # max_IP to reduce the stack of arrays, each containing one labelled region, to a single 2D np array.
    final_array_labelled = np.sum(final_array, axis=0)

    return final_array_labelled


def dilate_labels(lbl_path=None, labels=None, dilate_iterations=1, save: bool = False, save_dir=None,
                  structuring_element=None) -> List:
    """
    Erodes given label images by a specified number of pixels (erosion_iterations)

    Parameters
    -----------
    lbl_path : str
        directory of a folder with label images
    dilate_iterations : int
        how many pixels erode from edges, int default = 1
    save : bool
        if save processed files
    save_dir : str
        directory where eroded label images are saved, default = labels_path + "/eroded"

    Returns
    -----------
    final_eroded_list : list
        a list of eroded labels images

    """
    logger = logging.getLogger(__name__)
    logger.info(f"Dilating labels from: {lbl_path}")

    if lbl_path is not None:
        labels, fnames = load_files(files_dir=lbl_path, return_fnames=True)
    else:
        fnames = [str(i) + ".tif" for i in enumerate(labels)]

    final_dilated_list = []

    for name, lbl in tqdm(zip(fnames, labels), total=len(labels), desc="Dilating labels"):
        mask = lbl.astype('int32')
        dilated_label = []

        for z_slice in range(mask.shape[0]):
            # if there are no labels at all in the 2D image, do not perform any erosion but add it to a new list
            if mask[
                z_slice].max() == 0:  # add or mask[z_slice].max() == 1 if to not erode where there is just one label
                dilated_label.append(mask[z_slice])
                continue

            # erode all masks in one z-slice by one pixel
            # add eroded image to a new list
            eroded = _dilate_labels(mask[z_slice], dilate_iterations, structuring_element)
            dilated_label.append(eroded)

        # convert list of numpy arrays to stacked numpy array to get a 3D image again with eroded masks
        final_eroded = np.stack(dilated_label)

        final_dilated_list.append(final_eroded)

        if save:
            if save_dir is None:
                save_dir = os.path.join(lbl_path, "dilated")

            if not os.path.exists(save_dir):
                Path(save_dir).mkdir(parents=True, exist_ok=True)

            imwrite(os.path.join(save_dir, name), final_eroded)

    logger.info(f"Dilated labels saved to: {save_dir}")

    return final_dilated_list


def _binarise(i):
    if i > 1:
        return 1
    else:
        return i


def make_binary_labels(lbl_path, save: bool = True, save_path=None, return_final_list=False) -> List:
    """
    Binarises given label image from the path

    Parameters
    -----------
    lbl_path : str
        directory of a folder with label images
    save : bool
        if save processed files
    save_path :  str
        directory where eroded label images are saved

    Returns
    -----------
    binary_labels : list
        a list of binary labels images

    """
    logger = logging.getLogger(__name__)
    logger.info(f"Binarizing labels from: {lbl_path}")
    labels, fnames = load_files(lbl_path, return_fnames=True)

    if save:
        if save_path is None:
            save_path = os.path.join(lbl_path, "binary")
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)

        if fnames is not None:
            assert len(fnames) == len(labels), "The length of filenames list must be same as the length of images list"

    if return_final_list:
        binary_labels = []

    for labels, fname in tqdm(zip(labels, fnames), desc="Binarizing labels", total=len(labels)):
        func = np.vectorize(_binarise)
        binary_label = func(labels)
        if return_final_list:
            binary_labels.append(binary_label)

        if save:
            imwrite(os.path.join(save_path, fname), binary_label)
    if save:
        logger.info(f"Binary labels saved to: {save_path}")

    if return_final_list:
        return binary_labels


def split_into_train_and_validation(X, Y):
    """
    Utilities function from Stardist 3D training notebook to split given
    images and labels datasets into training and validation sets.

    Parameters
    -----------
    X : list
        a list of raw images
    Y : list
        a list of ground truth images

    Returns
    -----------
    X_trn, Y_trn, X_val, Y_val

    """

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.3 * len(ind)))) - 1
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]

    X_val, Y_val = [X[i] for i in ind_val], [np.array(Y[i]).astype("int32") for i in tqdm(ind_val, desc="Splitting into X_val and Y_val")]
    X_trn, Y_trn = [X[i] for i in ind_train], [np.array(Y[i]).astype("int32") for i in tqdm(ind_train, desc="Splitting into X_trn and Y_trn")]

    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    return X_trn, Y_trn, X_val, Y_val


def split_into_train_val_test(raw_dir=None, labels_dir=None, ratio=None, base_dir=None):
    """
    Splits one original folder into training, validation and test folders. If no validation folder is needed ratio
    should be a list of two items.

    data/
        train/
            raw_timepoints/
                0.tif
                ...
            labels_timepoints/
                0.tif
                ...
        val/
            raw_timepoints/
                0.tif
                ...
            labels_timepoints/
                0.tif
                ...
        test/
            raw_timepoints/
                0.tif
                ...
            labels_timepoints/
                0.tif
                ...

    """
    import splitfolders

    if ratio is None:
        ratio = [0.7, 0.2, 0.1]

    # organize data to folders suitable for using splitfolders module
    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copytree(raw_dir, os.path.join(tmpdirname, "raw"))
        shutil.copytree(labels_dir, os.path.join(tmpdirname, "labels"))
        output_path = base_dir
        splitfolders.ratio(tmpdirname, output=output_path, seed=1337, ratio=ratio, group_prefix=None, move=True)


# fixme: remove files from dir where files should be copied to because it keeps raising an error
def _copy(source, destination, filename):
    logger = logging.getLogger(__name__)

    if not os.path.exists(destination):
        Path(destination).mkdir(parents=True, exist_ok=True)
        logger.info(f"{destination} directory was created")

    fsource = os.path.join(source, filename)
    fdestination = os.path.join(destination, filename)
    logger.info(f"File {fsource} is being copied to {fdestination}...")
    copyfile(fsource, fdestination)
    logger.info(f"Copying finished")


def copy_data(source, original_dir, name):
    logger = logging.getLogger(__name__)
    files = os.listdir(source)  # get the name of files in the directory
    logger.info(f"Files found in the {source} directory: {files}")

    file = files[0]
    fsource = os.path.join(source, file)
    image = imread(fsource)

    if len(files) == 1 and len(image.shape) == 4:
        destination = os.path.join(original_dir, name + "_4D")
        _copy(source, destination, file)

        return destination
    elif len(files) >= 1 and len(image.shape) == 3:
        destination = os.path.join(original_dir, name + "_timepoints")
        for f in files:
            _copy(source, destination, f)

        return destination
    else:
        logger.info("In this project, only multiple 3D or one 4D file are supported as original input files. If you"
                    "have such files make sure that path variables in the .env file or paths given as parameters when"
                    "running src.data.make_dataset module contain correct files.")


def split_into_timepoints(filedir, save_dir, prefix=""):
    logger = logging.getLogger(__name__)
    filenames = os.listdir(filedir)
    assert (len(filenames) == 1)
    filename = filenames[0]

    image = imread(os.path.join(filedir, filename))
    logger.info(f"Original image shape: {image.shape}")

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"{save_dir} directory was created")

    logger.info("Splitting 4D image into multiple 3D images...")
    for t, timepoint in enumerate(tqdm(range(image.shape[0]))):
        img = image[timepoint]
        imwrite(os.path.join(save_dir, prefix + str(t) + '.tif'), img)


def fill_small_holes(Y, save_dir: str = None):
    """
    Utilities function to fill small holes in labels using a function from Stardist.

    Parameters
    -----------
    Y : list
        a list of label images
    save_dir : str
        directory where normalized images will be saved

    Returns
    -----------
    Y
        a list of label images with filled in small holes.

    """
    from stardist import fill_label_holes

    logger = logging.getLogger(__name__)
    logger.info("Filling small holes in labels...")

    Y = [fill_label_holes(y.astype("int32")) for y in tqdm(Y)]

    if save_dir is not None:
        logger.info("Saving label images with filled small holes...")
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for i, y in enumerate(Y):
            imwrite(os.path.join(save_dir, str(i) + ".tif"), y)
        logger.info("Saving label images finished")

    return Y


def normalize_data(X, n_channel=1, save_dir: str = None):
    """
    Utilities function to normalize the data using a function from csbdeep library.

    Parameters
    -----------
    X : list
        a list of raw images
    n_channel : int
        number of channels
    save_dir : str
        directory where normalized images will be saved

    Returns
    -----------
    X
        a list of normalized images

    """
    from csbdeep.utils import normalize

    logger = logging.getLogger(__name__)
    assert n_channel == 1, "Unfortunately only single channel images are supported in this repository for now. Please" \
                           "feel free to modify forked repository to tailor it to your dataset"

    logger.info("Normalizing images...")
    X = [normalize(x, 1, 99.8, axis=(0, 1, 2)) for x in tqdm(X)]

    if save_dir is not None:
        logger.info("Saving normalized images...")
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for i, x in enumerate(X):
            imwrite(os.path.join(save_dir, str(i) + ".tif"), x)
        logger.info("Saving finished")

    return X


def make_patches_from_image(image, patches_size=None, step=64, cfg=None, return_shape=False):
    """
    Make patches from 3D image using library patchify to generate training datasets that fit into memory.

    Parameters
    ---------
    image: np.array
        an image to make patches from
    patches_size: tuple
        a tuple of patch sizes in z, y, x or provided in the config
    step: int
        overlap between patches or provided in the config
        The required condition for unpatchify to success is to have (width - patch_width) mod step_size = 0
    cfg: YACS config
        configuration for the experiment
    return_shape:
        return the shape after patchify, which is need to run unpatchify() after the prediction on patches is done

    Returns
    ---------
    patches : nd.array
        array of patches (nr of patches, z, y, x).
        (optional) also, shape of an array after patchifying to use with the unpatchify()
    """
    from patchify import patchify, unpatchify

    shape = image.shape
    x = np.array(image)
    print(f"Original img shape: {shape}")

    if cfg is not None:
        patches_size = cfg.MODEL.TRAIN.PATCHES_SIZE
        step = cfg.MODEL.TRAIN.PATCHES_STEP

        if cfg.DATASET.EXPAND_DIM:
            # add additional 2D images to a stack to have axes dividable by 4 because it was needed for some tools
            x = np.vstack([image, np.reshape(np.zeros(shape[1] * shape[2] * cfg.DATASET.EXPAND_SIZE),
                                             (cfg.DATASET.EXPAND_SIZE, shape[1], shape[2]))])
            expanded_shape = x.shape
            print(f"Dataset expanded to shape: {x.shape}")
    assert patches_size is not None, "Missing patches size"

    patchified_img = patchify(x, patches_size, step=step)
    print(f"Patchified image shape: {patchified_img.shape}")
    reconstructed_image = unpatchify(patchified_img, x.shape)
    assert (
            reconstructed_image == x).all(), f"Cannot reconstruct the image with original shape {shape} if " \
                                             f"{patches_size}, step={step} are used"

    #  The required condition for unpatchify to success is to have (width - patch_width) mod step_size = 0
    # assert (len(images[0][2]) - patches_size[2]) % step == 0

    # reshape patches to: n_patches, x, y, z
    reshaped = np.reshape(patchified_img, (-1, patchified_img.shape[3],
                                           patchified_img.shape[4],
                                           patchified_img.shape[5]))
    if return_shape and cfg is not None:
        if cfg.DATASET.EXPAND_DIM:
            return reshaped, patchified_img.shape, expanded_shape
    elif return_shape:
        return reshaped, patchified_img.shape
    else:
        return reshaped


def make_patches_from_list_of_images(images, labels, patches_size=None, step=64, cfg=None):
    """
    Make patches using library patchify to generate training datasets that fit into memory.

    Parameters
    -----------
    images: list
        a list of images from which to generate patches
    labels: list
        a list of labels from which to generate patches
    patches_size: tuple
        a tuple of patch sizes in z, y, x or provided in the config
    step: int
        overlap between patches or provided in the config
        The required condition for unpatchify to success is to have (width - patch_width) mod step_size = 0
    cfg: YACS config
        configuration for the experiment
    Returns
    ---------
    img_patches, labels_patches
        a list of patches of images and a list of patches of labels
    """
    from patchify import patchify, unpatchify

    # creating new lists where image and labels patches will be stored
    img_patches = []
    labels_patches = []

    if cfg is not None:
        # get defined patches size and patches step
        patches_size = cfg.MODEL.TRAIN.PATCHES_SIZE
        print(f"Patches size from cfg: {cfg.MODEL.TRAIN.PATCHES_SIZE}")
        step = cfg.MODEL.TRAIN.PATCHES_STEP
        print(f"Patches step from cfg: {cfg.MODEL.TRAIN.PATCHES_STEP}")

    assert patches_size is not None

    for x, y in tqdm(zip(images, labels), total=len(labels), desc="Making patches"):

        if cfg is not None and cfg.DATASET.EXPAND_DIM:
            # this was used in case it is needed for some tools to have specific size images
            x = np.vstack([x, np.reshape(np.zeros(x.shape[1] * x.shape[2] * cfg.DATASET.EXPAND_SIZE),
                                         (cfg.DATASET.EXPAND_SIZE, x.shape[1], x.shape[2]))])
            y = np.vstack([y, np.reshape(np.zeros(y.shape[1] * y.shape[2] * cfg.DATASET.EXPAND_SIZE),
                                         (cfg.DATASET.EXPAND_SIZE, y.shape[1], y.shape[2]))])

        patchified_x = patchify(x, patches_size, step=step)
        patchified_y = patchify(y, patches_size, step=step)

        # the required condition for unpatchify to success is to have (width - patch_width) mod step_size = 0
        reconstructed_image = unpatchify(patchified_x, x.shape)
        assert (reconstructed_image == x).all(), f"Cannot reconstruct the image to original with shape {x.shape} " \
                                                 f"if {patches_size}, step={step} are used"
        reconstructed_image = unpatchify(patchified_y, y.shape)
        assert (reconstructed_image == y).all(), f"Cannot reconstruct the image to original with shape {y.shape} " \
                                                 f"if {patches_size}, step={step} are used"

        # reshape to: n_patches, z, y, x
        reshaped_x = np.reshape(patchified_x, (-1, patchified_x.shape[3],
                                               patchified_x.shape[4],
                                               patchified_x.shape[5]))

        reshaped_y = np.reshape(patchified_y, (-1, patchified_y.shape[3],
                                               patchified_y.shape[4],
                                               patchified_y.shape[5]))
        patches_img_list = [reshaped_x[i, :, :, :] for i in range(reshaped_x.shape[0])]
        patches_labels_list = [reshaped_y[i, :, :, :] for i in range(reshaped_y.shape[0])]
        for img_patch, lbl_patch in zip(patches_img_list, patches_labels_list):
            img_patches.append(img_patch)  # append each patch to the list
            labels_patches.append(lbl_patch)

    print(f"Img patches[0] shape: {img_patches[0].shape}")
    print(f"Total nr of patches: {len(img_patches)}")

    print(f"Labels patches[0] shape: {labels_patches[0].shape}")
    print(f"Total nr of labels patches: {len(labels_patches)}")

    return img_patches, labels_patches


def convert_tif_to_hdf5(trn_raw_dir, trn_labels_dir, val_raw_dir, val_labels_dir, save_dir):
    import h5py

    # MAKE TRAIN DATASET
    (images, fnames), (labels, _) = load_dataset(rawdir=trn_raw_dir, labelsdir=trn_labels_dir, return_fnames=True)
    print(fnames)
    hdf5_fnames = [name.split(".")[0] + ".hdf5" for name in fnames]
    print(hdf5_fnames)

    Path(os.path.join(save_dir, "train")).mkdir(parents=True, exist_ok=True)

    for img, lbl, name in tqdm(zip(images, labels, hdf5_fnames), desc="Converting .tif files to hdf5. Train dataset",
                               total=len(images)):
        hf = h5py.File(os.path.join(save_dir, "train", name), 'a')  # open a hdf5 file
        hf.create_dataset(name="raw", data=np.array(img))  # write raw data to hdf5 file
        hf.create_dataset(name="label", data=np.array(lbl))  # write label data to hdf5 file
        hf.close()  # close the hdf5 file
    del images, labels, fnames, hdf5_fnames

    # MAKE VALIDATION DATASET
    (images, fnames), (labels, _) = load_dataset(rawdir=val_raw_dir, labelsdir=val_labels_dir, return_fnames=True)
    hdf5_fnames = [name.split(".")[0] + ".hdf5" for name in fnames]

    Path(os.path.join(save_dir, "val")).mkdir(parents=True, exist_ok=True)

    for img, lbl, name in tqdm(zip(images, labels, hdf5_fnames), desc="Converting .tif files to hdf5. Val dataset",
                               total=len(images)):
        hf = h5py.File(os.path.join(save_dir, "val", name), 'a')  # open a hdf5 file
        hf.create_dataset(name="raw", data=np.array(img))  # write raw data to hdf5 file
        hf.create_dataset(name="label", data=np.array(lbl))  # write label data to hdf5 file
        hf.close()  # close the hdf5 file


def _binarize_slow(labels: List) -> List:
    """
    Makes a binary labels list for multiple labels images.
    Slow, exchanged by func::make_binary_labels
    """
    binary_labels = []

    for labels_img in tqdm(labels):

        # initiate a new array for a binary labels image of the same shape as original labels img
        array_zyx = np.empty(shape=labels_img.shape, dtype=int)

        # iterate though all elements of an array (of one labels image)
        for z, z_stack in enumerate(labels_img):
            for y, y_arr in enumerate(z_stack):
                for x, x_arr in enumerate(y_arr):
                    # if an array element is already 1 or 0 (background) keep it in the new array
                    if x_arr == 0 or x_arr == 1:
                        array_zyx[z][y][x] = x_arr
                    elif x_arr > 1:  # else exchange it for 1 in the new (binary) array
                        array_zyx[z][y][x] = 1
                    else:
                        print("Error. Labels image contains negative values..")

        binary_labels.append(array_zyx)

    return binary_labels


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

