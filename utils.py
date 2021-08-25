import json
import logging
import pickle
import os
import datetime
import numpy as np


logging.basicConfig(level=logging.INFO, format="'%(asctime)s - %(name)s: %(levelname)s: %(message)s'")
Logger = logging.getLogger("UTILS.STDOUT")
Logger.setLevel("INFO")

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
        Logger.error("Json data loading Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        Logger.info("Json data of {} loaded successfully.".format(AbsolutePath))
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
        Logger.error("Json data writing Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        Logger.info("Json data of {} wrote successfully.".format(AbsolutePath))
        File.close()

def import_from_pkl(AbsolutePath, FileName=None):
    try:
        AbsolutePath = os.path.join(AbsolutePath, FileName) if FileName else AbsolutePath
        File = open(AbsolutePath, 'rb')
        Content = pickle.load(File)
    except Exception as e:
        Logger.error("Pickle data loading Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        Logger.info("Pickle data of {} loaded successfully.".format(AbsolutePath))
        File.close()
        return Content

def export_to_pkl(AbsolutePath, FileName=None, Content=None):
    try:
        AbsolutePath = os.path.join(AbsolutePath, FileName) if FileName is not None else AbsolutePath
        if (isinstance(Content, set)):
            Content = list(Content)
        File = open(AbsolutePath, 'wb')
        pickle.dump(Content, File)
    except Exception as e:
        Logger.error("Pickle data writing Failed with exception: {}.".format(e))
        if "File" in dir():
            File.close()
    else:
        Logger.info("Pickle data of {} wrote successfully.".format(AbsolutePath))
        File.close()

def get_absolute_path(Path):
    try:
        AbsolutePath = os.path.abspath(os.path.join(os.path.dirname(__file__), Path))
    except Exception as e:
        Logger.error("Getting absolute path for {} Failed with exception {}.".format(Path, e))
    else:
        return AbsolutePath

def preprocess_data():
    good_sample_path = './middle_data/good_sample'
    mal_sample_path = './middle_data/mal_sample'

    if not os.path.exists(good_sample_path):
        os.makedirs(good_sample_path)
    if not os.path.exists(mal_sample_path):
        os.makedirs(mal_sample_path)

    sample_list = import_from_json('./raw_data', 'apgx.json')
    label_list = import_from_json('./raw_data', 'apgy.json')
    meta_list = import_from_json('./raw_data', 'apgm.json')

    feature_list = import_from_pkl('./middle_data', 'feature_list.data')

    for i, sample in enumerate(sample_list):
        name = meta_list[i]['sha256'] + '.feature'
        sample = set(sample.keys())
        features = []
        for feature in feature_list:
            if feature in sample:
                features.append(1)
            else:
                features.append(0)
        if label_list[i] == 1:
            export_to_pkl(mal_sample_path, name, features)
        else:
            export_to_pkl(good_sample_path, name, features)

def format_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")


