import logging
import utils
import feature_extract
import time
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

logging.basicConfig(level=logging.INFO, format="'%(asctime)s - %(name)s: %(levelname)s: %(message)s'")
Logger = logging.getLogger("CLASSIFY.STDOUT")
Logger.setLevel("INFO")


def Train(SamplePath, TrainSize, Model=None):
    """ collect and split Samples. """
    Samples, Labels, Features = feature_extract.LoadData(utils.GetAbsolutePath(SamplePath))

    """ generate feature space. """
    Logger.info("Generating feature space...")
    # mlb = MultiLabelBinarizer(sparse_output=True)
    mlb = MultiLabelBinarizer()
    x = mlb.fit_transform(Samples)
    y = np.array(Labels)
    y[y == 0] = -1 # malware will be marked as 1 otherwise will be marked as -1.
    Logger.info("Generated Feature Space Done.")

    """ split samples to training set and test set. """
    Logger.info("Splitting samples to train-test set...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=TrainSize, random_state=10)
    Logger.info("Train-test Split Done.")

    """ train the models with SVM. """
    Logger.info("Start Training Classifier with SVM.")
    """
    if not Model:
        Classifier = GridSearchCV(LinearSVC(), Params, cv=5, scoring='f1', n_jobs=-1, verbose=2)
        # Classifier = GridSearchCV(LinearSVC(loss='hinge'), Params, cv=5, scoring='f1', n_jobs=-1, verbose=2)
        # Classifier = GridSearchCV(LinearSVC(penalty='l1', dual=False), Params, cv=5, scoring='f1', n_jobs=-1, verbose=2)
        # Classifier = GridSearchCV(SVC(), parameters, cv=5, scoring='f1', n_jobs=-1, verbose=2)
        Models = Classifier.fit(x_train, y_train)
        BestModel = Models.best_estimator_
        File = "model_" + utils.FormatTime()
        joblib.dump(Classifier, File + '.pkl')
    else:
        Models = joblib.load(Model)
        BestModel = Models.best_estimator_

    Report = Evaluate(BestModel, x_test, y_test)
    print(Report)
    """
    Params = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
    parameters = [{'C': [0.01, 0.1, 1, 10, 100],
                   'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                   'gamma': [100, 10, 1, 0.1, 0.01, 0.001],
                   'degree': [2, 3, 4, 5]}]
    Classifiers = [
        GridSearchCV(LinearSVC(), Params, cv=5, scoring='f1', n_jobs=-1, verbose=2),
        GridSearchCV(SVC(), parameters, cv=5, scoring='f1', n_jobs=-1, verbose=2)
    ]
    for i, cls in enumerate(Classifiers):
        Models = cls.fit(x_train, y_train)
        BestModel = Models.best_estimator_
        File = "svm_model_" + utils.FormatTime()
        joblib.dump(cls, File + '.pkl')
        Report = Evaluate(BestModel, x_test, y_test)
        with open("Report_" + str(i), "w") as f:
            f.write(Report)
        del Models, BestModel, File, Report


def Evaluate(Model, x_test, y_test):
    """ evaluate the models on test set """
    Time = time.time()
    y_pred = Model.predict(x_test)
    Accuracy = accuracy_score(y_test, y_pred)
    TPR = recall_score(y_test, y_pred)
    FPR = 1 - recall_score(y_test, y_pred, pos_label=-1)  # FPR = 1 - TNR and TNR = specificity

    Report = "Test Set Report:\n" \
             "Accuracy = {}.\n" \
             "TPR = {}.\n" \
             "FPR = {}.\n" \
             "{}".format(Accuracy, TPR, FPR,
                         classification_report(y_test, y_pred, labels=[1, -1], target_names=['Malware', 'Goodware']))

    return Report


def ExportTopNFeat():
    pass

if __name__ == '__main__':
    Train('./updated_data', 0.7)