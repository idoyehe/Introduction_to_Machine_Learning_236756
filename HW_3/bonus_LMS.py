from data_infrastructure import *
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, SGDRegressor
import numpy as np


def one_vs_all(clf, x_train, y_train, class_index):
    y_train_modified = y_train.astype('int').copy()
    y_train_modified = np.vectorize(lambda t: 1 if t == class_index else -1)(y_train_modified)
    return clf.fit(x_train, y_train_modified)


class LMS:
    """ LMS class that implements Widrow-Hoff algorithm """

    def __init__(self, eta, max_iterations, mse_no_change):
        self.mse_no_change = mse_no_change
        self.eta = eta
        self.max_iterations = max_iterations
        self.weights = None

    def fit(self, x, y):
        # initialize weights and bias
        self._init_weights_vector((x.shape[1], 1))
        prev_mse = 0
        mse_no_change_counter = 0
        y_pred = np.zeros(y.shape)
        new_mse = self.calc_mse(y, y_pred)
        for iteration in range(self.max_iterations):
            random_sample_index = np.random.choice(x.shape[0], replace=False)
            x_i = x[random_sample_index].reshape((x.shape[1], 1))
            y_pred_i = self._calc_y_pred(x_i)
            y_pred[random_sample_index] = y_pred_i
            new_mse = self.calc_mse(y, y_pred)
            mse_no_change_counter += new_mse == prev_mse
            if mse_no_change_counter == self.mse_no_change:
                return self, iteration - mse_no_change_counter, new_mse
            prev_mse = new_mse
            self.weights += self.eta * (y[random_sample_index] - y_pred_i) * x_i

        return self, self.max_iterations, new_mse

    def _calc_y_pred(self, x_i):
        scalar = self.weights.T @ x_i
        return scalar[0][0]

    def _init_weights_vector(self, weights_shape):
        self.weights = np.zeros(weights_shape, dtype='float64')

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
    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


def load_digits_dataset(test_size=0.15):
    digits = load_digits()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    x, y = digits.data, digits.target
    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


def clf_evaluation(clf, clf_title, x_test, y_test, class_index):
    y_pred = np.sign(clf.predict(x_test))
    y_test_modified = y_test.astype('int').copy()
    y_test_modified = np.vectorize(lambda t: 1 if t == class_index else -1)(y_test_modified)
    print(f"{clf_title} accuracy score: {100 * accuracy_score(y_pred=y_pred, y_true=y_test_modified)}%")


def main():
    digits_x_train, digits_x_test, digits_y_train, digits_y_test = load_digits_dataset()
    iris_x_train, iris_x_test, iris_y_train, iris_y_test = load_iris_dataset()

    print("Iris Dataset Compare")
    max_iterations = 100
    print(f"Max iterations {max_iterations}")
    lms_clf = LMS(eta=0.001, max_iterations=max_iterations, mse_no_change=5)
    perceptron_clf = Perceptron(random_state=0, max_iter=max_iterations, alpha=0.001, tol=0.001)

    for curr_class in range(3):
        print(f"current class {curr_class}")
        lms_clf, iteration, new_mse = one_vs_all(lms_clf, np.append(iris_x_train, np.ones((iris_x_train.shape[0],1)), axis=1), iris_y_train, curr_class)
        if iteration < max_iterations:
            print(f"LMS cover in {iteration} iteration")
        clf_evaluation(lms_clf, "LMS", np.append(iris_x_test, np.ones((iris_x_test.shape[0],1)), axis=1), iris_y_test, curr_class)

        perceptron_clf = one_vs_all(perceptron_clf, iris_x_train, iris_y_train, curr_class)
        clf_evaluation(perceptron_clf, "Perceptron", iris_x_test, iris_y_test, curr_class)

    print("\nDigits Dataset Compare\n")
    for curr_class in range(10):
        print(f"current class {curr_class}")
        lms_clf, iteration, new_mse = one_vs_all(lms_clf, np.append(digits_x_train, np.ones((digits_x_train.shape[0],1)), axis=1), digits_y_train, curr_class)
        if iteration < max_iterations:
            print(f"LMS cover in {iteration} iteration")
        clf_evaluation(lms_clf, "LMS", np.append(digits_x_test, np.ones((digits_x_test.shape[0],1)), axis=1), digits_y_test, curr_class)

        perceptron_clf = one_vs_all(perceptron_clf, digits_x_train, digits_y_train, curr_class)
        clf_evaluation(perceptron_clf, "Perceptron", digits_x_test, digits_y_test, curr_class)


if __name__ == '__main__':
    np.random.seed(0)
    main()
