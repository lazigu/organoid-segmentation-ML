from pathlib import Path

import apoc
import logging
import os
import pandas as pd
from tqdm import tqdm

from src.config.config import combine_cfgs
from src.data.preprocess_utils import load_files
from tifffile import imwrite, imread
import pyclesperanto_prototype as cle

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cle.select_device("RTX")


def apoc_train(exp_dir, image_folder, masks_folder, cfg):
    """
    Trains apoc classifier on folders

    Parameters
    ----------
    exp_dir : str
        base directory of the experiment
    image_folder : str
        a path to the folder containing images
    masks_folder : str
        a path to the folder containing labels
    cfg : CfgNode
        configuration, which contains:
        clf : str
            type of the classifier: ProbabilityMapper, ObjectSegmenter, ObjectClassifier
        features : str
            a string of space separated features, e.g. "gaussian_blur=1 top_hat_box=20"
        cl_filename : str
            a path where trained classifier will be written (contains the name of the classifier and extension .cl)
        max_depth : int
            depth of trees
        num_ensembles : int
            number of trees
        output_probability_of_class : int
        positive_class_identifier : int

    """

    logger = logging.getLogger(__name__)
    clf = None

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    cl_path = os.path.join(exp_dir, cfg.APOC.CL_FILENAME)
    logger.info(f"Classifier path: {cl_path}")

    with open(os.path.join(exp_dir, cfg.APOC.CL_FILENAME.split(".")[0] + ".txt"), 'w') as f:
        f.write(cfg.APOC.FEATURES)
        logger.info(f"Features were written to: {os.path.join(exp_dir, cfg.APOC.CL_FILENAME.split('.')[0] + '.txt')}")

    apoc.erase_classifier(cl_path)

    if cfg.APOC.CLF == "ProbabilityMapper":
        clf = apoc.ProbabilityMapper(opencl_filename=cl_path, max_depth=cfg.APOC.MAX_DEPTH,
                                     num_ensembles=cfg.APOC.NUM_ENSEMBLES,
                                     output_probability_of_class=cfg.APOC.OUTPUT_PROBABILITY_OF_CLASS)
    elif cfg.APOC.CLF == "ObjectSegmenter":
        clf = apoc.ObjectSegmenter(opencl_filename=cl_path, max_depth=cfg.APOC.MAX_DEPTH,
                                   num_ensembles=cfg.APOC.NUM_ENSEMBLES,
                                   positive_class_identifier=cfg.APOC.POSITIVE_CLASS_IDENTIFIER)
    elif cfg.APOC.CLF == "ObjectClassifier":
        clf = apoc.ObjectClassifier(cl_path)

    # apoc needs that a path ends with / so check if that is the case
    if not image_folder.endswith("/"):
        image_folder = image_folder + "/"
    if not masks_folder.endswith("/"):
        masks_folder = masks_folder + "/"

    # train classifier on folders
    if clf is not None:
        logger.info("Training the classifier...")
        apoc.train_classifier_from_image_folders(
            clf,
            cfg.APOC.FEATURES,
            image=image_folder,
            ground_truth=masks_folder)
        logger.info("Training finished")

    stats_path = os.path.splitext(cl_path)[0] + ".csv"

    stats, _ = clf.statistics()
    df = pd.DataFrame(stats)
    swapped = df.swapaxes("index", "columns")
    swapped["sum"] = swapped.sum(axis=1)
    swapped.to_csv(stats_path)
    logger.info(f"Statistics of the classifier were saved to: {stats_path}")


def apoc_predict(filespath, exp_dir, cfg=None, cfg_path=None, classifier_dir=None):
    """
    Performs predictions with apoc classifier and saves them to the created results directory. If classifier dir is
    None, cl file is loaded from exp_dir.

    Returns
    ----------
    None
    """
    logger = logging.getLogger(__name__)
    if cfg is None and cfg_path is not None:
        cfg = combine_cfgs(cfg_path)
    elif cfg is None and cfg_path is None:
        logger.info("Please provide configuration node or a path to YAML file!")

    results_dir = os.path.join(exp_dir, "results", cfg.APOC.CL_FILENAME + "_results")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if classifier_dir is None:
        classifier_dir = exp_dir
        cl_path = os.path.join(exp_dir, cfg.APOC.CL_FILENAME)
        logger.info(f"Classifier path: {cl_path}")
    else:
        cl_path = os.path.join(classifier_dir, cfg.APOC.CL_FILENAME)

    clf = None

    if cfg.APOC.CLF == "ProbabilityMapper":
        clf = apoc.ProbabilityMapper(opencl_filename=cl_path)
    elif cfg.APOC.CLF == "ObjectSegmenter":
        clf = apoc.ObjectSegmenter(opencl_filename=cl_path)
    elif cfg.APOC.CLF == "ObjectClassifier":
        clf = apoc.ObjectClassifier(opencl_filename=cl_path)

    features_file = cfg.APOC.CL_FILENAME.split(".")[0] + ".txt"
    with open(os.path.join(classifier_dir, features_file)) as f:
        features = f.read()
        print(features)

    if features != cfg.APOC.FEATURES:
        logger.info(f"Different features found in {features_file} and CFG file \n {features} \n {cfg.APOC.FEATURES}) "
                    f"Features from CFG file will be overwritten with features from {features_file}")

    _, fnames = load_files(filespath, return_fnames=True)

    if clf is not None:
        logger.info("Prediction has started...")

        for fname in tqdm(fnames):
            img = imread(os.path.join(filespath, fname))
            labels = clf.predict(img, features=features)
            imwrite(os.path.join(results_dir, fname), labels)

        logger.info("Predictions finished")
    else:
        logger.info("Classifier has not been loaded successfully, prediction cannot be performed.")

    logger.info(f"Predictions were saved to: {results_dir}")


def visualise_decision_tree(tree_dir):
    """
    Based on [1].
    .dot file containing a graph to visualise a tree must be obtained when training a random forest classifier in the
    following way:

    model = RandomForestClassifier(n_estimators=10)
    model.fit(data, gt)

    # Extract single tree
    estimator = model.estimators_[5]

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names = feature_names,
                    class_names = gt_names,
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    [1] https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
    """

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', tree_dir, '-o', tree_dir.split(".")[0] + ".png", '-Gdpi=600'])

    # Display in jupyter notebook
    # from IPython.display import Image
    # Image(filename=tree_dir.split(".")[0] + ".png")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
