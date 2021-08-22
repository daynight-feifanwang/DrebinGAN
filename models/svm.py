import os
import sys
import logging
import joblib
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(os.path.split(os.path.dirname(__file__))[0])
import utils

logging.basicConfig(level=logging.INFO, format="'%(asctime)s - %(name)s: %(levelname)s: %(message)s'")
logger = logging.getLogger("DrebinSVM.STDOUT")
logger.setLevel("INFO")


class DrebinSVM(object):
    def __init__(self, load_path='./updated_data', save_path='./save_model', train_size=0.7):
        self.load_path = utils.get_absolute_path(load_path)
        self.save_path = utils.get_absolute_path(save_path)
        self.train_size = train_size
        self.raw_data_path = utils.get_absolute_path('./raw_data')
        self.updated_data_path = utils.get_absolute_path('./updated_data')
        self.raw_samples = 'apg-X.json'
        self.raw_labels = 'apg-Y.json'
        self.clean_threshold = 3
        self.url_allowed = True

        self.model = None
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.feature_list = None

    def train(self):
        ###### collect samples #####
        logger.info("Loading data for DrebinSVM...")

        samples, labels, features = self.load_feature()

        logger.info("Loading Done.")

        ##### generate feature space #####
        logger.info("Generating feature space...")
        """
        mal_samples = [sample for i, sample in enumerate(samples) if labels[i] == 1]
        good_samples = [sample for i, sample in enumerate(samples) if labels[i] == 0]

        batch = round(len(mal_samples) * 4)
        new_good_samples = []

        for i in range(len(good_samples) // batch):
            new_good_samples.append(good_samples[i * batch : (i + 1) * batch])
        """
        if self.feature_list:
            x = self.mlb.transform(samples)
        else:
            x = self.mlb.fit_transform(samples)
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
        parameters = [{'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                       'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                       'gamma': [100, 10, 1, 0.1, 0.01, 0.001],
                       'degree': [2, 3, 4, 5]}]
        # classifier = GridSearchCV(LinearSVC(loss='hinge'), parameters, cv=5, scoring='f1', n_jobs=-1, verbose=2)
        self.model = GridSearchCV(SVC(probability=True), parameters, cv=5, scoring='f1', n_jobs=-1, verbose=5)

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

        with open("../Report_DrebinSVM", "w") as f:
            f.write(report)

    def predict(self, x=None):
        if self.model is None:
            self.load()

        model = self.model.best_estimator_
        self.mlb.fit(self.feature_list)

        x = self.load_data(x)
        y = model.predict(x)
        z = model.predict_proba(x)
        return np.array(y), np.array(z)

    def load(self):
        self.model = joblib.load(os.path.join(self.save_path, 'svm_model.pkl'))

        feature_list = utils.import_from_pkl(self.updated_data_path, 'total_feature_list.data')
        self.feature_list = [[feature] for feature in feature_list]

    def save(self, classifier):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        joblib.dump(classifier, os.path.join(self.save_path, 'svm_model.pkl'))

    def transform(self, x):
        return x

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
        logger.info("Loading Data...")
        if 'sample_list.data' in os.listdir(load_abs_path):
            samples = utils.import_from_pkl(load_abs_path, 'sample_list.data')
        else:
            # Extract samples' features from raw data.
            samples = utils.import_from_json(self.raw_data_path, self.raw_samples)
            samples, updated_features = self.clean_data(samples)  # Format samples with clean[optional].
            utils.export_to_pkl(self.updated_data_path, 'sample_list.data', samples)
        logger.info("Loading samples DONE.")

        if 'label_list.data' in os.listdir(load_abs_path):
            labels = utils.import_from_pkl(load_abs_path, 'label_list.data')
        else:
            # Extract samples' labels from raw data.
            labels = utils.import_from_json(self.raw_data_path, self.raw_labels)
            labels = [1 if label == 1 else -1 for label in labels]
            utils.export_to_pkl(self.updated_data_path, 'label_list.data', labels)
        logger.info("Loading labels DONE.")

        if 'total_feature_list.data' in os.listdir(load_abs_path):
            features = utils.import_from_pkl(load_abs_path, 'total_feature_list.data')
            if 'updated_features' in dir():
                features = sorted(updated_features)
                utils.export_to_pkl(self.updated_data_path, 'total_feature_list.data', features)

        else:
            # Extract using features from samples.
            features = sorted(updated_features)
            utils.export_to_pkl(self.updated_data_path, 'total_feature_list.data', features)
        logger.info("Loading Feature vocabulary Done.")

        return samples, labels, features

    def clean_data(self, samples):
        # url_allowed == True -> ignore threshold for url
        # raw sample type: list[dict[str, int]] -> samples[sample[feature, 1]]
        try:
            samples = [list(sample.keys()) for sample in samples]
        except Exception as e:
            logger.debug("Formatting raw features FAILED with exception {}.".format(e))
        data = [features for sample in samples for features in sample]
        features_count = Counter(data)
        features = set()
        for feat, count in features_count.items():
            if self.url_allowed and feat.startswith('urls'):
                features.add(feat)
            elif count >= self.clean_threshold:
                features.add(feat)
            else:
                continue
        for i_s, sample in enumerate(samples):
            samples[i_s] = [feature for feature in sample if feature in features]

        return samples, list(features)


if __name__ == '__main__':
    c = DrebinSVM()
    c.train()
    """
    x_real = utils.import_from_pkl('../updated_data/sample_list.data')
    y_real = utils.import_from_pkl('../updated_data/label_list.data')
    os.rename('../updated_data/sample_list.data', '../updated_data/sample_list.data_')
    os.rename('../updated_data/label_list.data', '../updated_data/label_list.data_')

    y_pred, z_pred = c.predict('./updated_data/sample_list.data_')

    x_updated = [x for i, x in enumerate(x_real) if y_real[i] == 1]
    y_updated = [1 for i in range(len(x_updated))]

    x_good = []
    for i, _ in enumerate(x_real):
        if y_real[i] == -1 and y_pred[i] == -1:
            tmp = x_real[i]
            tmp.append(z_pred[i][0] - z_pred[i][1])
            x_good.append(tmp)
    x_good.sort(key=lambda x: x[-1])
    x_good = [x[:-1] for x in x_good]
    x_updated += x_good[:round(len(x_good) * 0.6)]

    y_updated += [-1 for i in range(round(len(x_good) * 0.6))]

    utils.export_to_pkl('../updated_data/sample_list.data', Content=x_updated)
    utils.export_to_pkl('../updated_data/label_list.data', Content=y_updated)

    c.train()

    x = c.mlb.transform(x_real)
    y = y_real
    c.evaluate(x, y)

    os.remove('../updated_data/sample_list.data')
    os.remove('../updated_data/label_list.data')
    os.rename('../updated_data/sample_list.data_', '../updated_data/sample_list.data')
    os.rename('../updated_data/label_list.data_', '../updated_data/label_list.data')
    """

