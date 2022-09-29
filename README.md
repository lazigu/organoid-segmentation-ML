# Organoid Segmentation: Classical Machine Learning & Deep Learning

This repository contains the Supplementary Material for the master thesis on segmentation of early mESC organoid 4D 
(3D+time) data (not publicly available yet). There are scripts to train and use for prediction deep learning models and 
classical machine learning (random forest) classifiers for the segmentation. Furthermore, the repository contains data
preprocessing, postprocessing and segmentation evaluation functions.
The following tools are used in this repository to evaluate the potential for early organoid segmentation:
* [Accelerated Pixel and Object Classification (APOC)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification)
* [PlantSeg](https://github.com/hci-unihd/plant-seg)
* [StarDist](https://github.com/stardist/stardist)
* [VollSeg](https://github.com/Kapoorlabs-CAPED/VollSeg)
* [Local minima seeded watershed](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes)
* Using StarDist prediction directly as seeds for watershed to improve segmentation at boundaries

### Getting started

To start applying segmentation tools and their evaluation from this repository, clone this repository and navigate to 
the project´s directory (../organoid-segmentation-ML).
```
git clone https://github.com/lazigu/organoid-segmentation-ML.git
cd organoid-segmentation-ML
```
Then, required packages could be installed using provided environment file, e.g., using conda. However, this might 
result in conflicts between packages due to heavy dependencies.

```
conda create --name org-seg --file requirements_tf.txt
```

### Pre-processing the data

To prepare data for training, open cloned repository with your favourite IDE and create <i>.env</i> file (optional), where you 
should define data directories for PATH_DATA_RAW and PATH_LABELS variables. This file is added to <i>.gitignore</i>, 
therefore, it is not pushed to GitHub and stays private. However, this step is optional, and instead you can 
always run all the following commands with given paths in the command itself.
Create <i>"make_dataset.YAML"</i> file in the <i>src/config/</i> directory. In the created 
file, define parameters in the DATASET node that should differ from defaults defined in <i>src/config/config.py</i>. 
Directories should contain either one 4D file per raw or labels data folder or multiple 3D images. Other data shapes are 
unfortunately not supported in this repository.
Then, run the command:

```
python -m src.data.make_dataset -cfg src/config/make_dataset.yaml <RAW_PATH (optional)> <LABELS_PATH (optional)>
```

Data is processed and new folders are created in such structure, depending on the pre-processing options chosen. Choose
unique CFG.DATASET.NAME and/or CFG.DATASET.PREPROCESS_NAME if you do not wish old files to be overwritten. Processed 
folder also gets a date identifier in the name automatically, so delete old folders manually if your data is big/old 
files are not needed anymore.

    data/
        DATASET_NAME_original/
                            labels_4D/
                                    4D_original_file.tif
                            raw_4D/
                                    4D_labels_file.tif
                            raw_timepoints/
                                    0.tif
                                    ...
                            labels_timepoints/
                                    0.tif
                                    ...
        CFG_NAME_preprocessed_DATE_IDENTIFIER/
                            images/
                                isotropic/ (optional)
                                    0.tif
                                    ...
                                    scaled/ (optional)
                                        0.tif
                                        ...
                                        normalized/ (optional)
                                                0.tif
                                                ...
                            labels/
                                isotropic/ (optional)
                                    0.tif
                                    ...
                                    eroded/ (optional)
                                        0.tif
                                        ...
                                        binary/ (optional)
                                            0.tif
                                            ...
                                            scaled/ (optional)
                                                    0.tif
                                                    ...
                                                filled_holes/ (optional)
                                                        0.tif
                                                        ...

### Training

To train a model create a YAML file in the src/config/experiments directory and specify training parameters that should 
be different from the default parameters in src/config/config.py and run:

```
python -m src.models.train_model -cfg src/config/experiments/expXX.yaml data/CFG_NAME_train_val_test/train/image_timepoints data/CFG_NAME_train_val_test/train/labels_timepoints
```

where XX is your experiment´s number. Trained model, training configuration and tensorboard logs will be saved to 
`../organoid-segmentation-ML/experiments/expXX`.

For specifying training data directories in the command line and not .env file, enter the following command for help:

```
python -m src.models.train_model --help
```

While a model is training you can track the progress in TensorBoard, which was already installed when creating an 
environment in the first step. To activate TensorBoard, navigate to the experiment´s folder to see where the  logs are 
created and stored, e.g. experiments/expXX/stardist/logs. Then, run the command below with a specified logs directory:

```
tensorboard --logdir logs
```

### Prediction

To use already trained model for segmentation, create a YAML file in the src/config/experiments directory and specify 
prediction parameters that should be different from the default parameters in src/config/config.py. Most importantly,
a classifier´s head name (stardist, vollseg, plantseg, apoc) needs to be specified there, and pre-trained model´s name 
and directory if it is saved elsewhere and not in the experiment´s directory. Then, run the command, where XX is your 
experiment´s number:

```
python -m src.models.predict_model -cfg src/config/experiments/expXX.yaml
```
