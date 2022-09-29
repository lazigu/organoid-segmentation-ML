import os
import shutil

import napari
from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle
from tensorflow.python.client import device_lib
import seaborn as sns


def copy_model_to_backup(model_src, exp_dir):
    backup_dir = os.path.join(exp_dir, "model_backups")
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    shutil.copytree(model_src, backup_dir)
    print(f"A copy of the model has been made from {model_src} to {backup_dir}")


def visualize_prob_maps_napari(test_images, segmenter, features):
    viewer = napari.Viewer()

    for i, filename in enumerate(test_images):
        image = imread(filename)
        viewer.add_image(image, name=str(i), contrast_limits=[np.array(image).min(), np.array(image).max()])

        labels = segmenter.predict(image, features=features)

        viewer.add_image(labels, name=str(i), colormap="plasma",
                         contrast_limits=[np.array(labels).min(), np.array(labels).max()])


def visualize_feature_stack(feature_stack, features, generate_plots=False, open_in_napari=False):
    """
    Generates matplotlib plots for each feature in the stack.
    Visualizes only the middle z-slice. Also opens napari with all feature images for 3D exploration.

    Parameters
    -----------
    feature_stack : features stack generated by APOC
    features : a string of features separated by a space
    generate_plots : use Matplotlib to generate plots
    open_in_napari : open feature images in napari

    Returns
    -----------
    None
    """

    features_specs = features.split(" ")

    if open_in_napari:
        import napari
        viewer = napari.Viewer()
        for name, f in zip(features_specs, feature_stack):
            viewer.add_image(f, name=name, contrast_limits=[np.array(f).min(), np.array(f).max()])

    if generate_plots:
        z_slice = feature_stack.shape[0] // 2
        if len(feature_stack) == 0:
            return

        width = 3
        height = int(len(feature_stack) / 3)
        if height * width < len(feature_stack):
            height = height + 1

        fig, axes = plt.subplots(height, width, figsize=(10, 10))

        for i, f in enumerate(feature_stack):
            if height > 1:
                cle.imshow(f[z_slice], plot=axes[int(i / 3)][i % 3], colormap=plt.cm.gray)
            else:
                cle.imshow(f[z_slice], plot=axes[i], colormap=plt.cm.gray)

        w = len(feature_stack) % width
        if w > 0:
            w = width - w
            while w > 0:
                if height > 1:
                    axes[-1][width - w].set_visible(False)
                else:
                    axes[width - w].set_visible(False)
                w = w - 1

        plt.show()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """
    Function from Stardist
    Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y
