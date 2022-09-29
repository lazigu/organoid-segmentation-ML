import json
import os
from pathlib import Path
import yaml


def load_json(path):
    assert path.endswith('.json'), "Whoops that´s not a json file"
    with open(path, 'r') as file:
        return json.load(file)


def write_json(data, path):
    with open(path, 'w') as file:
        file.write(json.dumps(data))


def write_yaml(data, path):
    with open(path, "w") as file:
        yaml.dump(data, file)


def load_yaml(path):
    assert path.endswith('.yaml'), "Whoops that´s not a yaml file"
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def get_tensorboard_logs(path_to_events_file_train, tag):
    """
    Utilities function to get Tensorboard logs with the specified tag from the given path
    """
    import tensorflow as tf
    from tensorflow.python.summary.summary_iterator import summary_iterator

    value_list = []
    for e in summary_iterator(path_to_events_file_train):
        for v in e.summary.value:
            # print(v.tag)
            if v.tag == tag:

                value = tf.make_ndarray(v.tensor)
                value_list.append(value)

    return value_list


def save_stardist_matching_named_tuple_metrics(save_path, metrics, taus):
    """
    Input must be a list of DatasetMatching named tuples for different tau values.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for m, tau in zip(metrics, taus):
        write_json(m._asdict(), os.path.join(save_path, "metrics_tau_" + str(tau).replace(".", ",") + ".json"))


def load_stardist_matching_named_tuple_metrics(taus, path):
    """
    Parameters
    -----------
    taus : list
        a list of different tau values for which matching dataset statistics had been calculated.
    path : str
        relative path where multiple json files or each tau value are located, e.g. "metrics/exp12". Each file must
        consist of a dictionary with keys: "criterion", "thresh", "fp", "tp", "fn", "precision", "recall", "accuracy",
        "f1", "n_true", "n_pred", "mean_true_score", "mean_matched_score", "panoptic_quality", by_image"
    """

    from collections import namedtuple

    def load(fpath):
        assert fpath.endswith('.json'), "Whoops that´s not a json file"
        with open(fpath, 'r') as file:
            dict_loaded = json.load(file)
            return DatasetMatching(**dict_loaded)

    metrics_return = []
    DatasetMatching = namedtuple('DatasetMatching',
                                 ["criterion", "thresh", "fp", "tp", "fn", "precision", "recall", "accuracy", "f1",
                                  "n_true", "n_pred", "mean_true_score", "mean_matched_score", "panoptic_quality",
                                  "by_image"])
    for tau in taus:
        metrics_return.append(load(os.path.join(path, "metrics_tau_" + str(tau).replace(".", ",") + ".json")))

    return metrics_return
