from data_infrastructure import *
from sklearn.datasets import load_iris, load_digits, make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt


def one_vs_all(clf, x_train, y_train, class_index, x_val=None, y_val=None):
    y_train_modified = y_train.astype('int').copy()
    y_train_modified = np.vectorize(lambda t: 1 if t == class_index else -1)(y_train_modified)
    if x_val is None or y_val is None:
        return clf.fit(x_train, y_train_modified)
    y_val_modified = y_val.astype('int').copy()
    y_val_modified = np.vectorize(lambda t: 1 if t == class_index else -1)(y_val_modified)
    return clf.fit(x_train, y_train_modified, x_val, y_val_modified)


class LMS:
    """ LMS class that implements Widrow-Hoff algorithm """

    def __init__(self, eta, max_epochs: int = None):
        self.eta = eta
        self.max_epochs = max_epochs
        self.weights = None
        self.train_stats = None
        self.valid_stats = None
        self.weights_stats = None
        self.best_epoch = None

    def fit(self, x_train, y_train, x_val, y_val):
        # initialize weights and bias
        self._init_weights_vector((x_train.shape[1], 1))
        self.train_stats = []
        self.valid_stats = []
        self.weights_stats = []
        self.best_epoch = None
        for epoch in range(self.max_epochs):
            for xi, target in zip(x_train, y_train):
                y_pred_i = self._calc_y_pred(xi.reshape((x_train.shape[1], 1)))
                self.weights += self.eta * (target - y_pred_i) * xi.reshape((x_train.shape[1], 1))

            self.train_stats.append(1 - accuracy_score(y_true=y_train, y_pred=self.predict(x_train)))
            self.valid_stats.append(1 - accuracy_score(y_true=y_val, y_pred=self.predict(x_val)))
            self.weights_stats.append(self.weights.copy())

        self.best_epoch = self.valid_stats.index(min(self.valid_stats))
        self.weights = self.weights_stats[self.best_epoch]
        return self

    def _calc_y_pred(self, x_i):
        scalar = self.weights.T @ x_i
        return scalar[0][0]

    def _init_weights_vector(self, weights_shape):
        self.weights = np.random.uniform(-1, 1, weights_shape)

    def calc_mse(self, y, y_pred):
        mse_error = y - y_pred
        mse_error = (mse_error ** 2).sum() / 2.0
        return mse_error

    def predict(self, x):
        return np.sign(x @ self.weights)


def load_iris_dataset(test_size=0.15):
    iris = load_iris()
    sss = StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=test_size)
    x, y = iris.data, iris.target
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


def load_digits_dataset(test_size=0.15):
    digits = load_digits()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    x, y = digits.data, digits.target
    x = np.delete(x, 0, axis=1)
    x = np.delete(x, 31, axis=1)
    x = np.delete(x, 37, axis=1)
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()
    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


def clf_evaluation(clf, clf_title, x_test, y_test, class_index):
    y_pred = np.sign(clf.predict(x_test))
    y_test_modified = y_test.astype('int').copy()
    y_test_modified = np.vectorize(lambda t: 1 if t == class_index else -1)(y_test_modified)
    print(f"{clf_title} accuracy score: {100 * accuracy_score(y_pred=y_pred, y_true=y_test_modified)}%")


def lms_vs_perceptron(load_dataset_function, dataset_name, classes, test_size=0.25):
    x_train, x_test, y_train, y_test = load_dataset_function(test_size)

    print(f"{dataset_name} Dataset Compare")
    lms_clf = LMS(eta=0.001, max_epochs=100)
    perceptron_clf = Perceptron(random_state=0, alpha=0.001)  # , verbose=True)

    def comparing(_curr_class):
        print(f"current class {_curr_class}")
        lms_clf_fitted = one_vs_all(lms_clf, np.append(x_train, np.ones((x_train.shape[0], 1)), axis=1),
                                    y_train, _curr_class, np.append(x_test, np.ones((x_test.shape[0], 1)), axis=1),
                                    y_test)
        print(f"LMS cover in {lms_clf_fitted.best_epoch} iteration")
        plt.plot(lms_clf_fitted.valid_stats, 'b', label="validation")
        plt.plot(lms_clf_fitted.train_stats, 'r', label="train")
        plt.xlabel("# Epochs")
        plt.ylabel("Validation Error rate")
        plt.legend()
        plt.show()

        clf_evaluation(lms_clf_fitted, "LMS", np.append(x_test, np.ones((x_test.shape[0], 1)), axis=1), y_test, _curr_class)

        perceptron_clf_fitted = one_vs_all(perceptron_clf, x_train, y_train, _curr_class)
        clf_evaluation(perceptron_clf_fitted, "Perceptron", x_test, y_test, _curr_class)

    if classes > 2:
        for curr_class in range(classes):
            comparing(curr_class)
    else:
        comparing(0)


def loading_dataset_for_lms(test_size=0.15):
    x, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=1,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        weights=None,
        flip_y=0.1,
        class_sep=0.000001,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=0)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    np.random.seed(0)
    # lms_vs_perceptron(load_iris_dataset, "Iris", 3)
    lms_vs_perceptron(load_digits_dataset, "Digits", 10)
    # lms_vs_perceptron(loading_dataset_for_lms, "LMS better", 2, lms_max_iter=5000000, perceptron_max_iter=12)
