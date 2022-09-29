from __future__ import print_function, unicode_literals, absolute_import, division

from pathlib import Path

import numpy as np
import logging

from csbdeep.utils import tf
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from patchify import unpatchify
from skimage.filters.thresholding import threshold_multiotsu
from skimage.morphology import remove_small_objects, remove_small_holes, area_closing
from tifffile import imwrite, imread
from tqdm import tqdm
from src.data.preprocess_utils import split_into_train_and_validation, make_binary_labels, load_files, \
    make_patches_from_image, preprocess, make_patches_from_list_of_images
from stardist import random_label_cmap, fill_label_holes
from src.data.preprocess_utils import load_dataset
import segmentation_models_3D as sm
from segmentation_models_3D.losses import bce_jaccard_loss
from segmentation_models_3D.metrics import iou_score
import keras
import tensorflow
import pyclesperanto_prototype as cle

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

keras.backend.set_image_data_format('channels_last')

np.random.seed(42)
lbl_cmap = random_label_cmap()


def unet_train(raw_filepath, labels_filepath, output_path, cfg):
    logger = logging.getLogger(__name__)

    gpus = tensorflow.config.list_logical_devices('GPU')
    strategy = tensorflow.distribute.MirroredStrategy(gpus,
                                                      cross_device_ops=tensorflow.distribute.HierarchicalCopyAllReduce())

    binary_labels_dir = os.path.join(output_path, "data", "binary_labels")

    # check if binary label´s folder already exists in experiments/expXX/data
    if not os.path.exists(binary_labels_dir) or len(os.listdir(binary_labels_dir)) == 0:
        make_binary_labels(labels_filepath, save=True, save_path=binary_labels_dir)
        labels_filepath = binary_labels_dir
    else:
        print(f"Found existing binary labels directory in the experiment´s folder. Data will be loaded from there. "
              f"{binary_labels_dir}")
        labels_filepath = binary_labels_dir

    # downsample the data
    if cfg.DATASET.RESCALE:
        raw_filepath, labels_filepath = preprocess(raw_filepath, labels_filepath, cfg, output_path)

    # load prepared for training images and labels
    images, labels = load_dataset(rawdir=raw_filepath, labelsdir=labels_filepath, step=cfg.DATASET.LOAD_STEP)

    X_patches, Y_patches = make_patches_from_list_of_images(images, labels, cfg)

    # free up memory
    del images
    del labels

    assert X_patches[0].shape[0] % 4 == 0, f"z axis needs to be dividable by 4, but {X_patches[0].shape[0]} is not"
    assert X_patches[0].shape[1] % 4 == 0, f"y axis needs to be dividable by 4, but {X_patches[0].shape[1]} is not"
    assert X_patches[0].shape[2] % 4 == 0, f"x axis needs to be dividable by 4, but {X_patches[0].shape[2]} is not"

    X_train, Y_train, X_val, Y_val = split_into_train_and_validation(X_patches, Y_patches)

    del X_patches
    del Y_patches

    # take only a subset of val images because it needs to divide by the batch size
    batch_size = cfg.MODEL.TRAIN.BATCH_SIZE
    limit_train = batch_size * (len(X_train) // batch_size)
    X_train = X_train[:limit_train]
    Y_train = Y_train[:limit_train]
    limit_val = batch_size * (len(X_val) // batch_size)
    X_val = X_val[:limit_val]
    Y_val = Y_val[:limit_val]

    # set some training parameters according to dataset size
    steps_per_epoch = len(X_train) / batch_size
    assert len(
        X_train) % batch_size == 0, f"Length of X_train ({len(X_train)}) needs to be dividable by the batch size {(batch_size)}"
    val_steps = len(X_val) / batch_size
    assert len(
        X_val) % batch_size == 0, f"Length of X_val ({len(X_val)}) needs to be dividable by the batch size {(batch_size)}"
    if steps_per_epoch != cfg.MODEL.TRAIN.STEPS_PER_EPOCH:
        logger.info(f"Calculated steps per epoch value ({steps_per_epoch}) is not the same as in the configuration file"
                    f" ({cfg.MODEL.TRAIN.STEPS_PER_EPOCH}). Please update it in the cfg!")

    logger.info(f"Final X_train length {len(X_train)}")
    logger.info(f"Final X_val length {len(X_val)}")
    logger.info(f"Final Y_train length {len(Y_train)}")
    logger.info(f"Final Y_val length {len(Y_val)}")

    # add channel as the last dim
    X_train = [np.expand_dims(x, axis=3) for x in tqdm(X_train, desc="Adding channel dim for X_train")]
    X_val = [np.expand_dims(x, axis=3) for x in tqdm(X_val, desc="Adding channel dim for X_val")]
    Y_train = [np.expand_dims(y, axis=3) for y in tqdm(Y_train, desc="Adding channel dim for Y_train")]
    Y_val = [np.expand_dims(y, axis=3) for y in tqdm(Y_val, desc="Adding channel dim for Y_val")]

    model_input_shape = X_train[0].shape

    logger.info(f"Final X_train shape {X_train[0].shape}")
    logger.info(f"Final X_val shape {X_val[0].shape}")
    logger.info(f"Final Y_train shape {Y_train[0].shape}")
    logger.info(f"Final Y_val shape {Y_val[0].shape}")

    # wrap data in Dataset objects if we want to use Mirrored strategy = to train on multiple gpus
    train_data = tensorflow.data.Dataset.from_tensor_slices((np.array(X_train), np.array(Y_train)))
    val_data = tensorflow.data.Dataset.from_tensor_slices((np.array(X_val), np.array(Y_val)))

    # free up memory
    del X_train
    del X_val
    del Y_train
    del Y_val

    # The batch size must now be set on the Dataset objects
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    # Disable AutoShard if we want to use Mirrored strategy = to train on multiple gpus
    options = tensorflow.data.Options()
    options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options)

    # define reduce learning rate on plateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)

    # define early stopping callback if val loss is not improving in defined nr (patience) of epochs
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    # define model
    with strategy.scope():
        model = sm.Unet(backbone_name='resnet34',
                        encoder_weights=None,
                        input_shape=model_input_shape,
                        classes=1)

    model.compile(tf.keras.optimizers.Adam(learning_rate=cfg.MODEL.TRAIN.LEARNING_RATE),
                  loss=bce_jaccard_loss,
                  metrics=[iou_score])

    # create a callback that saves the best model's weights
    cp_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_path, "UNET", "weights_best.hdf5"),
        save_best_only=True,
        save_weights_only=True, verbose=1)

    # create a callback that saves the model's weights every epoch
    cp_callback_now = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_path, "UNET", "weights_now.hdf5"),
        verbose=1,
        save_weights_only=True,
        save_freq="epoch")

    # create tensorboard callback
    tensorboard_callback = TensorBoard(log_dir=str(os.path.join(output_path, "UNET", "logs")), write_graph=False,
                                       profile_batch=0)

    # start training
    model.fit(
        train_data,
        epochs=cfg.MODEL.TRAIN.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        validation_data=val_data,
        callbacks=[cp_callback_best, cp_callback_now, tensorboard_callback, reduce_lr, early_stopping]
    )

    print(model.summary())

    model_json = model.to_json()
    with open(os.path.join(output_path, "UNET", "model_architecture.json"), "w") as json_file:
        json_file.write(model_json)

    model.save(os.path.join(output_path, "UNET"))


def unet_predict(test_filepath, cfg, exp_dir):
    """Performs prediction with a trained UNET model and saves predictions to ..experiments/expXX/results directory

    Parameters
    -----------
    test_filepath: str
        path to the folder with test images
    cfg: CfgNode
        combined configuration in the priority order provided:
        .env > YAML exp config > default config
    exp_dir: str
        path to experiment´s folder, e.g. src/experiments/expXX
    base_dir : str
        directory of the project

    Returns
    -----------
    None

    """

    logger = logging.getLogger(__name__)

    try:
        model = keras.models.load_model(os.path.join(exp_dir, "UNET"),
                                        custom_objects={'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
                                                        "iou_score": sm.metrics.iou_score})
    except OSError:
        print("Model will be loaded from the checkpoint")
        model = sm.Unet(backbone_name='resnet34',
                        encoder_weights=None,
                        input_shape=(64, 64, 64, 1),
                        classes=1)
        model.compile(tf.keras.optimizers.Adam(learning_rate=cfg.MODEL.TRAIN.LEARNING_RATE),
                      loss=bce_jaccard_loss,
                      metrics=[iou_score])
        model.load_weights(os.path.join(exp_dir, "UNET", "weights_best.ckpt")).expect_partial()

    # visualise the architecture of the model
    # from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file=r'model_plot.png', show_shapes=True, show_layer_names=True)

    unet_pred_dir = os.path.join(exp_dir, "results", "unet_pred")
    Path(unet_pred_dir).mkdir(exist_ok=True)

    logger.info(f"UNET predictions will be saved to {unet_pred_dir}")

    binary_save_path = os.path.join(unet_pred_dir, "binary")
    pstprocessed_save_path = os.path.join(unet_pred_dir, "postprocessed")
    pred_save_path = os.path.join(unet_pred_dir, "predictions")

    Path(pstprocessed_save_path).mkdir(parents=True, exist_ok=True)
    Path(binary_save_path).mkdir(parents=True, exist_ok=True)
    Path(pred_save_path).mkdir(parents=True, exist_ok=True)

    fnames = load_files(test_filepath, return_fnames=True, step=1, return_only_fnames=True)

    for name in tqdm(fnames, desc="UNET prediction"):
        img = imread(os.path.join(test_filepath, name))
        original_shape = img.shape
        resampled_shape = original_shape

        if cfg.DATASET.RESCALE:
            # downsample the image
            img = cle.resample(img,
                               factor_x=cfg.DATASET.SCALE_FACTOR[2],
                               factor_y=cfg.DATASET.SCALE_FACTOR[1],
                               factor_z=cfg.DATASET.SCALE_FACTOR[0],
                               linear_interpolation=True)
            resampled_shape = img.shape
        if cfg.DATASET.EXPAND_DIM:
            patches, shape_for_unpatchify, expanded_shape = make_patches_from_image(img, cfg, return_shape=True)
        else:
            patches, shape_for_unpatchify = make_patches_from_image(img, cfg, return_shape=True)

        # add channel as the last dim
        patches_final = np.expand_dims(patches, axis=4)

        predicted_patches = model.predict(patches_final)

        # reshape to the shape we had after patchifying
        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                (shape_for_unpatchify[0], shape_for_unpatchify[1],
                                                 shape_for_unpatchify[2], shape_for_unpatchify[3],
                                                 shape_for_unpatchify[4], shape_for_unpatchify[5]))

        # repatch individual patches into the original volume shape
        # print(predicted_patches_reshaped.shape)
        # print(shape)
        if cfg.DATASET.EXPAND_DIM:
            reconstructed_image = unpatchify(predicted_patches_reshaped, expanded_shape)
        else:
            reconstructed_image = unpatchify(predicted_patches_reshaped, resampled_shape)
        # print(reconstructed_image.shape)
        # print((reconstructed_image[:reconstructed_image.shape[0]-cfg.DATASET.EXPAND_SIZE if cfg.DATASET.EXPAND_DIM else None]).shape)

        # thresholds = threshold_multiotsu(reconstructed_image[reconstructed_image], classes=2)
        # print(f"Thresholds {thresholds}")
        # regions = np.digitize(reconstructed_image, bins=[thresholds])
        # binary = reconstructed_image > 0
        imwrite(os.path.join(pred_save_path, name), reconstructed_image)
        binary = np.where(reconstructed_image > 0.2, 1, 0)

        # imwrite(os.path.join(pred_save_path, name), reconstructed_image)
        imwrite(os.path.join(binary_save_path, name), binary)

        removed_small_objects = np.array([remove_small_objects(np.array(binary[i, :, :], dtype=bool),
                                                               connectivity=1,
                                                               min_size=100) for i in range(binary.shape[0])])

        final = np.array([remove_small_holes(np.array(removed_small_objects[i, :, :], dtype=bool),
                                             area_threshold=100) for i in range(removed_small_objects.shape[0])])

        final = np.array([area_closing(final[i, :, :],
                         area_threshold=200) for i in range(final.shape[0])])

        # eroded = erode_labels()

        # filled_holes = fill_label_holes(removed_small_objects.astype("int32"))

        # remove those z slices that were added to get required shape
        if cfg.DATASET.EXPAND_DIM:
            final = final[:resampled_shape[0], :, :]

        # upsample to get original size data
        if cfg.DATASET.RESCALE:
            final = cle.resample(final,
                                 factor_x=original_shape[2] / final.shape[2],
                                 factor_y=original_shape[1] / final.shape[1],
                                 factor_z=original_shape[0] / final.shape[0],
                                 linear_interpolation=True)
            logger.info(f"Image is scaled to: {final.shape}")

        logger.info(f"Final output binary image shape: {final.shape}")

        assert original_shape == final.shape, f"Shape of test images ({original_shape}) is not the same as for final output images ({final.shape})"
        imwrite(os.path.join(pstprocessed_save_path, name), cle.pull(final).astype('uint16'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
