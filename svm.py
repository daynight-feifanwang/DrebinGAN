import os
import logging
import joblib
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import GenericUnivariateSelect, SelectFromModel, chi2

import utils

logging.basicConfig(level=logging.INFO, format="'%(asctime)s - %(name)s: %(levelname)s: %(message)s'")
logger = logging.getLogger("DrebinSVM.STDOUT")
logger.setLevel("INFO")


class DrebinSVM(object):
    def __init__(self, load_path='./middle_data', save_path ='./middle_data', model_path='./save_model', train_size=0.7):
        self.load_path = utils.get_absolute_path(load_path)
        self.save_path = utils.get_absolute_path(save_path)
        self.model_path = utils.get_absolute_path(model_path)
        self.train_size = train_size

        self.raw_data_path = utils.get_absolute_path('./raw_data')
        self.raw_samples = 'apgx.json'
        self.raw_labels = 'apgy.json'

        self.model = None
        self.mlb = MultiLabelBinarizer(sparse_output=True)

    def train(self):
        ###### collect samples #####
        logger.info("Loading data for DrebinSVM...")

        samples, labels, features = self.load_feature()
        feature_list = [[feature] for feature in features]

        logger.info("Loading Done.")

        ##### generate feature space #####
        logger.info("Generating feature space...")

        self.mlb.fit(feature_list)
        x = self.mlb.transform(samples)
        y = np.array(labels)
        # malware will be marked as 1 otherwise will be marked as -1

        logger.info("Generating Done.")

        ##### split samples to train-test set #####
        logger.info("Splitting samples to train-test set...")

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=self.train_size, random_state=10)

        logger.info("Splitting Done.")

        ##### train the classifier with SVM #####
        logger.info("Start training classifier with SVM...")

        # ready precondition
        parameters = [{
            'C': [0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10, 100],
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge']
        }]
        self.model = GridSearchCV(LinearSVC(max_iter=5000), parameters, cv=5, scoring='f1', n_jobs=-1, verbose=2, error_score=0.0)
        # self.model = GridSearchCV(SVC(probability=True), parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=5)

        # train phrase
        self.model.fit(x_train, y_train)

        # evaluate phrase
        self.evaluate(x_test, y_test)

        # save model
        self.save(self.model)

        logger.info("Training Done.")

    def evaluate(self, x_test, y_test, model=None):
        if model is None:
            model = self.model.best_estimator_

        y_pred = model.predict(x_test)

        num_good_sample = len([x for x in y_test if x == -1])
        num_mal_sample = len([x for x in y_test if x == 1])

        accuracy = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        report = "Test Set Report: total {} samples, {} malsamples, {} goodsamples.\n" \
                 "Accuracy = {}.\n" \
                 "F1_score = {}.\n" \
                 "{}".format(num_good_sample + num_mal_sample, num_mal_sample, num_good_sample,
                             accuracy, f1score,
                             classification_report(y_test, y_pred, labels=[1, -1],
                                                   target_names=['Malware', 'Goodware']))

        with open("./Report_DrebinSVM", "w") as f:
            f.write(report)

    def predict(self, x=None):
        if self.model is None:
            self.load()

        model = self.model.best_estimator_

        x = self.load_data(x)
        y = model.predict(x)
        z = model.predict_proba(x)
        return np.array(y), np.array(z)

    def load(self):
        self.model = joblib.load(os.path.join(self.model_path, 'svm_model.pkl'))

        feature_list = utils.import_from_pkl(self.load_path, 'total_feature_list.data')
        feature_list = [[feature] for feature in feature_list]

        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.mlb.fit(feature_list)

    def save(self, classifier):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        joblib.dump(classifier, os.path.join(self.model_path, 'svm_model.pkl'))

    def load_data(self, x):
        if type(x) is str:
            x = utils.get_absolute_path(x)
            x = utils.import_from_pkl(x)
        else:
            x = self.transform(x)

        x = self.mlb.transform(x)
        return x

    def load_feature(self):
        load_abs_path = utils.get_absolute_path(self.load_path)
        if not os.path.exists(load_abs_path):
            os.makedirs(load_abs_path)

        logger.info("Loading Data...")
        if 'sample_list.data' in os.listdir(load_abs_path):
            samples = utils.import_from_pkl(load_abs_path, 'sample_list.data')
        else:
            # Extract samples' features from raw data.
            samples = utils.import_from_json(self.raw_data_path, self.raw_samples)
            samples = [list(sample.keys()) for sample in samples]
            utils.export_to_pkl(self.save_path, 'sample_list.data', samples)
        logger.info("Loading samples DONE.")

        if 'label_list.data' in os.listdir(load_abs_path):
            labels = utils.import_from_pkl(load_abs_path, 'label_list.data')
        else:
            # Extract samples' labels from raw data.
            labels = utils.import_from_json(self.raw_data_path, self.raw_labels)
            labels = [1 if label == 1 else -1 for label in labels]
            utils.export_to_pkl(self.save_path, 'label_list.data', labels)
        logger.info("Loading labels DONE.")

        if 'feature_list.data' in os.listdir(load_abs_path):
            features = utils.import_from_pkl(load_abs_path, 'feature_list.data')
        else:
            # Extract using features from samples.
            features = [feature for sample in samples for feature in sample]
            features = sorted(list(set(features)))
            utils.export_to_pkl(self.save_path, 'feature_list.data', features)
        logger.info("Loading Feature vocabulary Done.")

        return samples, labels, features

    def clean_data(self, samples, features):
        # clean samples data with selected features.
        feature_set = set(features)
        for i_s, sample in enumerate(samples):
            samples[i_s] = [feature for feature in sample if feature in feature_set]

        utils.export_to_pkl(self.save_path, 'sample_list.data', samples)

    def pre_select_feature(self, param, clean_sample=False):
        # load features
        samples, labels, features = self.load_feature()

        # transform samples to sparse matrix
        x = self.mlb.fit_transform(samples)
        y = np.array(labels)

        # select features
        selector = GenericUnivariateSelect(chi2, mode='percentile', param=param)
        x_new = selector.fit_transform(x, y)

        # map features
        mask = selector.get_support(True)
        features = list(np.array(features)[mask])
        utils.export_to_pkl(self.save_path, 'feature_list.data', features)

        if clean_sample is True:
            self.clean_data(samples, features)

    def select_feature(self):
        # load model
        if self.model is None:
            self.load()

        # load data
        samples, labels, _ = self.load_feature()

        ### test ###
        selector = SelectFromModel(self.model.best_estimator_, prefit=True)
        x_new = selector.transform(samples)

        x = self.mlb.transform(samples)
        y = np.array(labels)

        # select features
        # rfecv = RFECV(self.model.best_estimator_, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
        # rfecv.fit(x, y)

        # print(rfecv.n_features_to_select)
        # print(rfecv.ranking_)


if __name__ == '__main__':
    c = DrebinSVM()
    c.pre_select_feature(20)
    c.train()
    # c.select_feature()


