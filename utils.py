import json
import logging
import pickle
import os
import datetime
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="'%(asctime)s - %(name)s: %(levelname)s: %(message)s'")
logger = logging.getLogger("UTILS.STDOUT")
logger.setLevel("INFO")

def import_from_json(AbsolutePath, FileName):
    """
    Import data in the json file from the given path.

    :param AbsolutePath: absolute path of the json file
    :return Content: Content in the json file
    """
    try:
        AbsolutePath = os.path.join(AbsolutePath, FileName)
        File = open(AbsolutePath, 'rb')
        Content = json.load(File)
    except Exception as e:
        logger.error("Json data loading Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        logger.debug("Json data of {} loaded successfully.".format(AbsolutePath))
        File.close()
        return Content

def export_to_json(AbsolutePath, FileName, Content):
    try:
        AbsolutePath = os.path.join(AbsolutePath, FileName)
        if (isinstance(Content, set)):
            Content = list(Content)
        File = open(AbsolutePath, 'wb')
        json.dump(Content, File, indent=4, sort_keys=True)
    except Exception as e:
        logger.error("Json data writing Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        logger.info("Json data of {} wrote successfully.".format(AbsolutePath))
        File.close()

def import_from_pkl(AbsolutePath, FileName=None):
    try:
        AbsolutePath = os.path.join(AbsolutePath, FileName) if FileName else AbsolutePath
        File = open(AbsolutePath, 'rb')
        Content = pickle.load(File)
    except Exception as e:
        logger.error("Pickle data loading Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        logger.debug("Pickle data of {} loaded successfully.".format(AbsolutePath))
        File.close()
        return Content

def export_to_pkl(absolute_path, file_name=None, content=None):
    absolute_path = os.path.join(absolute_path, file_name) if file_name is not None else absolute_path
    try:
        if (isinstance(content, set)):
            content = list(content)
        file = open(absolute_path, 'wb')
        pickle.dump(content, file)
    except Exception as e:
        logger.error("Pickle data writing Failed with exception: {}.".format(e))
        if "file" in dir():
            file.close()
    else:
        logger.debug("Pickle data of {} wrote successfully.".format(absolute_path))
        file.close()

def get_absolute_path(Path):
    try:
        AbsolutePath = os.path.abspath(os.path.join(os.path.dirname(__file__), Path))
    except Exception as e:
        logger.error("Getting absolute path for {} Failed with exception {}.".format(Path, e))
    else:
        return AbsolutePath

def process_data(load_path='./middle_data', save_path='./final_data'):
    good_sample_path = os.path.join(save_path, 'good_sample')
    mal_sample_path = os.path.join(save_path, 'mal_sample')

    if not os.path.exists(good_sample_path):
        os.makedirs(good_sample_path)
    if not os.path.exists(mal_sample_path):
        os.makedirs(mal_sample_path)

    meta_list = import_from_json('./raw_data', 'apg-meta.json')
    sample_list = import_from_pkl(load_path, 'sample_list.data')
    label_list = import_from_pkl(load_path, 'label_list.data')

    benign_feature_list = import_from_pkl(save_path, 'benign_feature_list.data')
    total_feature_list = import_from_pkl(save_path, 'feature_list.data')

    benign_feature_list = [benign_feature_list]
    total_feature_list = [total_feature_list]

    tot_mlb = MultiLabelBinarizer()
    ben_mlb = MultiLabelBinarizer()

    _ = tot_mlb.fit(total_feature_list)
    _ = ben_mlb.fit(benign_feature_list)


    for i, sample in enumerate(sample_list):
        name = meta_list[i]['sha256'] + '.feature'
        if label_list[i] == 1: # mal sample -> total feature
            sample = tot_mlb.transform([sample])
            export_to_pkl(mal_sample_path, name, sample)
        else:
            sample = ben_mlb.transform([sample])
            export_to_pkl(good_sample_path, name, sample)

def format_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")

def consume_time(timestamp):
    timestamp = int(timestamp)
    if (timestamp // 3600 != 0):
        return "{:.2f}h".format(timestamp / 3600)
    elif timestamp // 60 != 0:
        return "{:.2f}min".format(timestamp / 60)
    return "{:.2f}s".format(timestamp)

def plot(title, vis, x, y, win=None):
    if win is None:
        win = vis.line(
            X=np.asarray([x]),
            Y=np.asarray([y]),
            opts=dict(title=title, xlabel='Batch', ylabel='Loss')
        )
    else:
        vis.line(X=np.asarray([x]), Y=np.asarray([y]),
                 win=win, update='append')
    return win


def log_sample(title, vis, text, win=None):
    if win is None:
        win = vis.text(text, opts=dict(title=title))
    else:
        vis.text(text, win=win, append=True)
    return win


def benign2total(load_path):
    abs_load_path = get_absolute_path(load_path)
    benign_feature_list = import_from_pkl(os.path.join(abs_load_path), 'benign_feature_list.data')
    total_feature_list = import_from_pkl(os.path.join(abs_load_path, 'feature_list.data'))

    feature_dict = {}
    for i, feature in enumerate(total_feature_list):
        feature_dict[feature] = i

    return_map = []
    for feature in benign_feature_list:
        return_map.append(feature_dict[feature])

    return return_map, len(total_feature_list)

def sparse_maximum(a, b):
    is_bigger = a - b
    is_bigger.data = np.where(is_bigger.data < 0, 1, 0)
    return a - a.multiply(is_bigger) + b.multiply(is_bigger)