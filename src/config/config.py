import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode as CfgNode

# YACS overwrite these settings using YAML
__C = CfgNode()

cfg = __C
__C.DESCRIPTION = "Default configuration"
__C.SELECT_GPU = ""                            # gpu name on which to run pyclespranto functions
__C.DATASET = CfgNode()
__C.DATASET.NAME = ""  # optional name of the dataset, only needed if instead of overwriting old datasets, you want to create new NAME_original folder, include _ in the name
__C.DATASET.CFG_NAME = ""                      # optional unique name (prefix) for the preprocessed data folder, folder name also has a date in it by default, e.g. if PREPROCESS_NAME = "_stardist", the folder will be named "01-06-2022_stardist_preprocessed"
__C.DATASET.AXES = "ZYX"
__C.DATASET.VOXEL_SIZE = [1, 1, 1]
__C.DATASET.SPLIT_TO_FRAMES = False            # whether to split timelapse data into multiple 3D files
__C.DATASET.MAKE_ISOTROPIC = False
__C.DATASET.RESCALE = False                    # scale down for smaller image size to fit into memory
__C.DATASET.SCALE_FACTOR = [1, 1, 1]
__C.DATASET.NORMALIZE = False
__C.DATASET.FILL_HOLES = False                 # fill small holes in label images
__C.DATASET.N_CHANNEL_IN = 1                   # number of channels of given input image
__C.DATASET.N_CHANNEL_OUT = 1
__C.DATASET.N_CLASSES = None                   # None or int: number of object classes to use for multi-class prediction (None disables it)
#__C.DATASET.ANISOTROPY = (float, float, float) # anisotropy of objects along each of the axes. Use ``None`` to disable only for (nearly) isotropic objects shapes. Also see ``stardist.utils.calculate_extents``.
__C.DATASET.MAKE_BINARY_LABELS = False
__C.DATASET.MAKE_ERODED_LABELS = False
__C.DATASET.EROSION_ITERATIONS = 1
__C.DATASET.RGB = False
__C.DATASET.SPLIT_FOLDERS = False              # split datasets into training, validation and test sets. Stardist and Vollseg do the splitting into train and val automatically, but need to split manually for testing, so that models during training would not see test data
__C.DATASET.SPLIT_RATIO = [0.99, 0, 0.01]
__C.DATASET.EXPAND_DIM = False                 # for training unet patches of equal size are need, e.g. (64, 64, 64), therefore, z axis should be expanded by np.zeroes if the shape in z is < 64
__C.DATASET.EXPAND_SIZE = 0
__C.DATASET.LOAD_STEP = 1

__C.DATASET.CROP = False                       # enable/disable cropping
__C.DATASET.CROP_SIZE = [318, 30, 71, 359, 66, 354]  # list items will be converted to [size[0]:size[1], size[2]:size[3], size[4]:size[5]] tp perform cropping of 3D np.array

__C.DATASET.PATCHES_SIZE = (64, 160, 160)
__C.DATASET.PATCHES_STEP = 64

# default directories where data is located. Overwritten with variables from .env
__C.DATASET.PATH_DATA_RAW = ""
__C.DATASET.PATH_LABELS = ""

# data augmentation parameters with albumentations library
# __C.DATASET.AUGMENTATION = CfgNode()

__C.MODEL = CfgNode()
__C.MODEL.CLASSIFIER_HEAD = ""  # accepted: stardist, vollseg, plantseg, apoc

# ------------------------------------------APOC PARAMETERS-------------------------------------------------------------

__C.APOC = CfgNode()
__C.APOC.CL_FILENAME = "Classifier.cl"
__C.APOC.CLF = ""   # ProbabilityMapper, ObjectSegmenter or ObjectClassifier
__C.APOC.FEATURES = ""
__C.APOC.MAX_DEPTH = 4
__C.APOC.NUM_ENSEMBLES = 100
__C.APOC.OUTPUT_PROBABILITY_OF_CLASS = 1   # for ProbMapper
__C.APOC.POSITIVE_CLASS_IDENTIFIER = 1

# -----------------------------------STARDIST AND VOLLSEG PARAMETERS----------------------------------------------------

__C.MODEL.BACKBONE = CfgNode()
__C.MODEL.BACKBONE.NAME = "UNET"  # UNET or RESNET: name of the neural network architecture to be used as a backbone

# -----------------------------------------------UNET-------------------------------------------------------------------
# todo axes in Config object are of the neural network
__C.MODEL.BACKBONE.UNET = CfgNode()
__C.MODEL.BACKBONE.UNET.INPUT_SHAPE = [None, None, None, 1]
__C.MODEL.BACKBONE.UNET.N_DEPTH = 2               # number of U-Net resolution levels (down/up-sampling layers). vollseg default: 3
__C.MODEL.BACKBONE.UNET.KERNEL_SIZE = 3 # (3, 3, 3)   # convolution kernel size for all (U-Net) convolution layers. Vollseg takes just one integer and makes a tuple: (kern_size, kern_size, kern_size)
__C.MODEL.BACKBONE.UNET.N_FILTER_BASE = 32        # number of convolution kernels (feature channels) for first U-Net layer. Doubled after each down-sampling layer. vollseg default: 48
__C.MODEL.BACKBONE.UNET.POOL = (2, 2, 2)          # maxpooling size for all (U-Net) convolution layers.
__C.MODEL.BACKBONE.UNET.NET_CONV_AFTER_UNET = 128 # number of filters of the extra convolution layer after U-Net (0 to disable).
__C.MODEL.BACKBONE.UNET.ACTIVATION = "relu"
__C.MODEL.BACKBONE.UNET.LAST_ACTIVATION = "relu"
__C.MODEL.BACKBONE.UNET.BATCH_NORM = False
__C.MODEL.BACKBONE.UNET.DROPOUT = 0.0

# -----------------------------------------------RESNET-----------------------------------------------------------------

__C.MODEL.BACKBONE.RESNET = CfgNode()
__C.MODEL.BACKBONE.RESNET.N_BLOCKS = 4                  # number of ResNet blocks
__C.MODEL.BACKBONE.RESNET.KERNEL_SIZE = (3, 3, 3)       # convolution kernel size for all ResNet blocks
__C.MODEL.BACKBONE.RESNET.KERNEL_INIT = "he_normal"
__C.MODEL.BACKBONE.RESNET.N_FILTER_BASE = 32            # number of convolution kernels (feature channels) for ResNet blocks. N is doubled after every downsampling, see ``grid``
__C.MODEL.BACKBONE.RESNET.N_CONV_PER_BLOCK = 3
__C.MODEL.BACKBONE.RESNET.ACTIVATION = "relu"
__C.MODEL.BACKBONE.RESNET.BATCH_NORM = False
__C.MODEL.BACKBONE.RESNET.NET_CONV_AFTER_RESNET = 128   #number of filters of the extra convolution layer after ResNet (0 to disable).

__C.MODEL.GRID = (1, 1, 1)   # subsampling factors (must be powers of 2) for each of the axes. Model will predict on a subsampled grid for increased efficiency and larger field of view. (grid_z, grid_y, grid_x) - grid_y an grid_x ned to be provided for vollseg
__C.MODEL.USE_GPU = True     # indicate that the data generator should use OpenCL to do computations on the GPU.

# ----------------------------------------------TRAINING----------------------------------------------------------------

__C.MODEL.TRAIN = CfgNode()
__C.MODEL.TRAIN.PATCHES_SIZE = (128, 128, 128)   # size of patches to be cropped from provided training images. Vollseg defaults: (16, 256, 256)
__C.MODEL.TRAIN.PATCHES_STEP = 64
__C.MODEL.TRAIN.BACKGROUND_REG = 1e-4    # regularizer to encourage distance predictions on background regions to be 0.
__C.MODEL.TRAIN.FOREGROUND_ONLY = 0.9    # fraction (0..1) of patches that will only be sampled from regions that contain foreground pixels.
__C.MODEL.TRAIN.SAMPLE_CACHE = True      # activate caching of valid patch regions for all training images (disable to save memory for large datasets)
__C.MODEL.TRAIN.DIST_LOSS = "mae"  # training loss for star-convex polygon distances ('mse' or 'mae').
__C.MODEL.TRAIN.LOSS_WEIGHTS = (1, 1)    # (1,1) if self.n_classes is None else (1,)*(self.n_classes+1): weights for losses relating to (probability, distance).
__C.MODEL.TRAIN.EPOCHS = 400             # number of training epochs.
__C.MODEL.TRAIN.STEPS_PER_EPOCH = 200    # number of parameter update steps per epoch. vollseg default: 160
__C.MODEL.TRAIN.LEARNING_RATE = 0.0001   # learning rate for training.
__C.MODEL.TRAIN.BATCH_SIZE = 1           # batch size for training. vollseg default: 4
__C.MODEL.TRAIN.TENSORBOARD = True       # enable TensorBoard for monitoring training progress.
__C.MODEL.TRAIN.N_VAL_PATCHES = None     # number of patches to be extracted from validation images (``None`` = one patch per image).
__C.MODEL.TRAIN.REDUCE_LR = CfgNode()
__C.MODEL.TRAIN.REDUCE_LR.STATE = True   # parameter of ReduceLROnPlateau_ callback; set to ``None`` to disable.
__C.MODEL.TRAIN.REDUCE_LR.FACTOR = 0.5
__C.MODEL.TRAIN.REDUCE_LR.PATIENCE = 40
__C.MODEL.TRAIN.REDUCE_LR.MIN_DELTA = 0
__C.MODEL.TRAIN.STARDIST_RAYS = 128      # rays_Base, int, or None. Ray factory (e.g. Ray_GoldenSpiral). If an integer then Ray_GoldenSpiral(rays) will be used

# ------------------------------------------PREDICT STARDIST-----------------------------------------------------------

# __C.MODEL.STARDIST.PROB_THRESHOLD = 0.5
# __C.MODEL.STARDIST.NMS_THRESHOLD = 0.4

# ------------------------------------------TRAINING VOLLSEG------------------------------------------------------------

# configs applicable only to training VollSeg
__C.MODEL.TRAIN.VOLLSEG = CfgNode()
__C.MODEL.TRAIN.VOLLSEG.VAL_RAW_DIR = ""        # needed only when training Vollseg
__C.MODEL.TRAIN.VOLLSEG.VAL_REAL_MASK_DIR = ""  # needed only when training Vollseg. For UNET training data is put as patches to .npz file according to the given validation split
# __C.MODEL.TRAIN.VOLLSEG.BINARY_MASK_DIR = ""
__C.MODEL.TRAIN.VOLLSEG.GENERATE_NPZ = True
__C.MODEL.TRAIN.VOLLSEG.VALIDATION_SPLIT = 0.01 # default in vollseg
__C.MODEL.TRAIN.VOLLSEG.DOWNSAMPLE_FACTOR = 1
# __C.MODEL.TRAIN.VOLLSEG.EROSION_ITERATIONS = 1
__C.MODEL.TRAIN.VOLLSEG.N_PATCHES_PER_IMAGE = 4

__C.MODEL.TRAIN.VOLLSEG.TRAIN_UNET = True
__C.MODEL.TRAIN.VOLLSEG.TRAIN_STAR = False    # is not implemented yet in this repo, stardist can be trained separately

# ------------------------------------------PREDICTION PARAMETERS-------------------------------------------------------

__C.MODEL.PREDICTION = CfgNode()

# -------------------------------------------VOLLSEG PREDICTION---------------------------------------------------------

__C.MODEL.PREDICTION.VOLLSEG = CfgNode()
__C.MODEL.PREDICTION.VOLLSEG.N_TILES = (1, 1, 1)     # adjust the number of tiles depending on the GPU used, tiling ensures that image tiles fit into memory
__C.MODEL.PREDICTION.VOLLSEG.MODEL_NAME = "model"
__C.MODEL.PREDICTION.VOLLSEG.SLICE_MERGE = True      # whether unet create labelling in 3D or slice by slice can be set by this parameter, if true it will merge neighbouring slices
__C.MODEL.PREDICTION.VOLLSEG.UNET_MODEL = None
__C.MODEL.PREDICTION.VOLLSEG.STAR_MODEL = None
__C.MODEL.PREDICTION.VOLLSEG.ROI_MODEL = None
__C.MODEL.PREDICTION.VOLLSEG.NOISE_MODEL = None
__C.MODEL.PREDICTION.VOLLSEG.PROB_THRESH = None       # for stardist
__C.MODEL.PREDICTION.VOLLSEG.NMS_THRESH = None        # for stardist
__C.MODEL.PREDICTION.VOLLSEG.MIN_SIZE_MASK = 100      # minimum size in pixels for the mask region, regions below this threshold would be removed
__C.MODEL.PREDICTION.VOLLSEG.MIN_SIZE = 100           # minimum size in pixels for the cells to be segmented
__C.MODEL.PREDICTION.VOLLSEG.MAX_SIZE = 10000000      # maximum size of the region, set this to veto regions above a certain size
__C.MODEL.PREDICTION.VOLLSEG.USE_PROBABILITY = True   # use probability map for stardist to perform watershedding or use distance map
__C.MODEL.PREDICTION.VOLLSEG.DONORMALIZE = True       # normalize images as in stardist
__C.MODEL.PREDICTION.VOLLSEG.LOWER_PERC = 1
__C.MODEL.PREDICTION.VOLLSEG.UPPER_PERC = 99.8
__C.MODEL.PREDICTION.VOLLSEG.DOUNET = True            # use UNET for binary mask (else denoised). If your Unet model is weak we will use the denoising model to obtain the semantic segmentation map, set this to False if this is the case else set it to TRUE if you are using Unet to obtain the semantic segmentation map.
__C.MODEL.PREDICTION.VOLLSEG.SEEDPOOL = True          # enable/disable seed pooling from unet and stardist, if disabled it will only take stardist seeds
__C.MODEL.PREDICTION.VOLLSEG.IOU_THRESHOLD = 0.3      # threshold linking in vollseg-napari


# ---------------------------------------PLANTSEG PREDICTION PARAMETERS-------------------------------------------------

# configs for Plant-seg prediction
__C.MODEL.PLANTSEG = CfgNode()
__C.MODEL.PLANTSEG.PREPROCESSING = CfgNode()
__C.MODEL.PLANTSEG.PREPROCESSING.ACTIVE = True                      # enable/disable preprocessing
__C.MODEL.PLANTSEG.PREPROCESSING.SAVE_DIRECTORY = "PreProcessing"   # create a new sub folder in the experiment folder where all results will be stored
__C.MODEL.PLANTSEG.PREPROCESSING.FACTOR = [1.0, 1.0, 1.0]           # Rescaling the volume is essential for the generalization of the networks. The rescaling factor can be computed as the resolution of the volume at hand divided by the resolution of the dataset used in training. Be careful, if the difference is too large check for a different model.
# the model was trained with data at voxel resolution of [0.235, 0.15, 0.15] (zxy micron). It is generally useful to rescale your input data to match the resolution of the original data # used in this work: [4.0, 1.0646153846153845, 1.0646153846153845]
__C.MODEL.PLANTSEG.PREPROCESSING.ORDER = 2                          # interpolation type (0 for nearest neighbors, 1 for linear spline, 2 for quadratic)
__C.MODEL.PLANTSEG.PREPROCESSING.CROP_VOLUME = "[:,:,:]"            # cropping out areas of little interest can drastically improve the performance of plantseg
__C.MODEL.PLANTSEG.PREPROCESSING.FILTER = CfgNode()                 # optional: perform Gaussian smoothing or median filtering on the input.
__C.MODEL.PLANTSEG.PREPROCESSING.FILTER.STATE = False               # enable/disable filtering
__C.MODEL.PLANTSEG.PREPROCESSING.FILTER.TYPE = "gaussian"           # accepted values: 'gaussian'/'median'
__C.MODEL.PLANTSEG.PREPROCESSING.FILTER.PARAM = 1.0                 # sigma (gaussian) or disc radius (median)

__C.MODEL.PLANTSEG.CNN = CfgNode()

__C.MODEL.PLANTSEG.CNN.PREDICTION = CfgNode()
__C.MODEL.PLANTSEG.CNN.PREDICTION.STATE = True                              # enable/disable UNet prediction
__C.MODEL.PLANTSEG.CNN.PREDICTION.MODEL_NAME = "generic_light_sheet_3d_unet"   # trained model name
__C.MODEL.PLANTSEG.CNN.PREDICTION.DEVICE = "cuda"                           # if a CUDA capable gpu is available and corrected setup use "cuda", if not you can use "cpu" for cpu only inference (slower)
__C.MODEL.PLANTSEG.CNN.PREDICTION.MIRROR_PADDING = [16, 32, 32]             # (int or tuple) mirror pad the input stack in each axis for best prediction performance
__C.MODEL.PLANTSEG.CNN.PREDICTION.PATCH_HALO = [8, 16, 16]                  # (int or tuple) padding to be removed from each axis in a given patch in order to avoid checkerboard artifacts
__C.MODEL.PLANTSEG.CNN.PREDICTION.NUM_WORKERS = 8                           # how many subprocesses to use for data loading
# __C.MODEL.PLANTSEG.CNN.PREDICTION.STRIDE = [20, 100, 100]                 # stride between patches (make sure that the patches overlap in order to get smoother prediction maps)
__C.MODEL.PLANTSEG.CNN.PREDICTION.STRIDE = "accurate"                       # accurate(slowest)/balanced/draft(fastest)
# __C.MODEL.PLANTSEG.CNN.PREDICTION.STRIDE_ACCURATE = 0.5
# __C.MODEL.PLANTSEG.CNN.PREDICTION.STRIDE_BALANCED = 0.75
# __C.MODEL.PLANTSEG.CNN.PREDICTION.STRIDE_DRAFT = 0.9
__C.MODEL.PLANTSEG.CNN.PREDICTION.VERSION = "best"                          # "best" refers to best performing on the val set (recommended), alternatively "last" refers to the last version before interruption
__C.MODEL.PLANTSEG.CNN.PREDICTION.MODEL_UPDATE = False                      # If "True" forces downloading networks from the online repos

__C.MODEL.PLANTSEG.CNN.POSTPROCESSING = CfgNode()
__C.MODEL.PLANTSEG.CNN.POSTPROCESSING.STATE = True                          # enable/disable cnn post processing
__C.MODEL.PLANTSEG.CNN.POSTPROCESSING.TIFF = True                           # if True convert to result to tiff
__C.MODEL.PLANTSEG.CNN.POSTPROCESSING.OUTPUT_TYPE = "data_float32"
__C.MODEL.PLANTSEG.CNN.POSTPROCESSING.RESCALING_FACTOR = [1, 1, 1]
__C.MODEL.PLANTSEG.CNN.POSTPROCESSING.SPLINE_ORDER = 2                      # spline order for rescaling (0 for nearest neighbors, 1 for linear spline, 2 for quadratic)
__C.MODEL.PLANTSEG.CNN.POSTPROCESSING.SAVE_RAW = False

__C.MODEL.PLANTSEG.SEGMENTATION = CfgNode()
__C.MODEL.PLANTSEG.SEGMENTATION.STATE = True                  # enable/disable segmentation
__C.MODEL.PLANTSEG.SEGMENTATION.NAME = "MultiCut"             # name of the algorithm to use for inferences. Options: MultiCut, MutexWS, GASP, DtWatershed
__C.MODEL.PLANTSEG.SEGMENTATION.BETA = 0.5                    # balance under-/over-segmentation; 0 - aim for undersegmentation, 1 - aim for oversegmentation. (Not active for DtWatershed)
__C.MODEL.PLANTSEG.SEGMENTATION.SAVE_DIRECTORY = "Segmentation"   # directory where to save the results
__C.MODEL.PLANTSEG.SEGMENTATION.RUN_WS = True                 # enable/disable watershed
__C.MODEL.PLANTSEG.SEGMENTATION.WS_2D = False                 # use 2D instead of 3D watershed
__C.MODEL.PLANTSEG.SEGMENTATION.WS_THRESHOLD = 0.5            # probability maps threshold. The CNN Predictions Threshold is used for the superpixels extraction and Distance Transform Watershed. It has a crucial role for the watershed seeds extraction and can be used similarly to the "Unde/Over segmentation factor" to bias the final result. A high value translate to less seeds being placed (more under segmentation), while with a low value more seeds are placed (more over segmentation).
__C.MODEL.PLANTSEG.SEGMENTATION.WS_MINSIZE = 50               # set the minimum superpixels size
__C.MODEL.PLANTSEG.SEGMENTATION.WS_SIGMA = 2.0                # sigma for the gaussian smoothing of the distance transform. If Watershed Seeds Sigma and Watershed Boundary Sigma are larger than zero a gaussian smoothing is applied on the input before the aforementioned operations. This is mainly helpful for the seeds computation, but in most cases does not impact segmentation quality.
__C.MODEL.PLANTSEG.SEGMENTATION.WS_W_SIGMA = 0.0                # sigma for the gaussian smoothing of boundary
__C.MODEL.PLANTSEG.SEGMENTATION.POST_MINSIZE = 50             # set the minimum segment size in the final segmentation. (Not active for DtWatershed)

__C.MODEL.PLANTSEG.SEGMENTATION.POSTPROCESSING = CfgNode()
__C.MODEL.PLANTSEG.SEGMENTATION.POSTPROCESSING.STATE = True  # enable/disable segmentation post processing
__C.MODEL.PLANTSEG.SEGMENTATION.POSTPROCESSING.TIFF = True   # if True convert to result to tiff
__C.MODEL.PLANTSEG.SEGMENTATION.POSTPROCESSING.RESCALING_FACTOR = [1, 1, 1]
__C.MODEL.PLANTSEG.SEGMENTATION.POSTPROCESSING.SPLINE_ORDER = 0   # spline order for rescaling (keep 0 for segmentation post processing)
__C.MODEL.PLANTSEG.SEGMENTATION.POSTPROCESSING.SAVE_RAW = False   # save raw input in the output segmentation file h5 file

# ---------------------------------SEEDED WATERSHED PARAMETERS----------------------------------------------------------

__C.WATERSHED = CfgNode()
__C.WATERSHED.SPOT_SIGMA = 5            # how close detected cells can be
__C.WATERSHED.OUTLINE_SIGMA = 2         # how precise segmented objects are outlined
__C.WATERSHED.MIN_INTENSITY = 600       # remove objects below this intensity threshold
__C.WATERSHED.CUSTOM_SEEDED_WT = False  # whether to do custom seeded watershed
__C.WATERSHED.PROPS_DIR = ""  # directory containing .csv files with measured region properties for each image (must be named same as images)
__C.WATERSHED.MASKS_DIR = ""  # directory containing binary masks files. Where mask == False, pixels will not be segmented. Only needed when custom seeded watershed is used


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return __C.clone()


def combine_cfgs(path_cfg_data: Path = None):
    """
    Combines configurations in the priority order provided: .env > YAML exp config > default config
    """
    # Priority 3: get default configs
    cfg_base = get_cfg_defaults()

    # Priority 2: merge from yaml config
    if path_cfg_data is not None and os.path.exists(path_cfg_data):
        cfg_base.merge_from_file(path_cfg_data)

    # Priority 1: merge from .env
    load_dotenv(find_dotenv(), verbose=True)  # Load .env

    # Load variables
    path_overwrite_keys = ['DATASET.PATH_DATA_RAW', os.getenv('PATH_DATA_RAW'),
                           'DATASET.PATH_LABELS', os.getenv('PATH_LABELS')]

    if path_overwrite_keys is not []:
        cfg_base.merge_from_list(path_overwrite_keys)

    return cfg_base
