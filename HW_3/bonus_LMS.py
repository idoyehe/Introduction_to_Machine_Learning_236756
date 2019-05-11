from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np


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
        for i in range(self.max_iterations):
            y_pred = self._calc_y_pred(x)
            new_mse = self.calc_mse(y, y_pred)
            mse_no_change_counter += new_mse == prev_mse
            if mse_no_change_counter == self.mse_no_change:
                return self, i - mse_no_change_counter, new_mse
            prev_mse = new_mse
            vect_error = (y - y_pred)
            self.weights += self.eta * x.T.dot(vect_error).reshape(self.weights.shape)

        return self, self.max_iterations, new_mse

    def _calc_y_pred(self, x):
        return (np.dot(x, self.weights)).flatten()

    def _init_weights_vector(self, weights_shape):
        self.weights = np.zeros(weights_shape, dtype='float64')

    def calc_mse(self, y, y_pred):
        mse_error = y - y_pred
        mse_error = (mse_error ** 2).sum() / 2.0
        return mse_error

    def predict(self, x):
        return np.where(self._calc_y_pred(x) < 0.0, 0, 1)


def load_iris_dataset(current_class, test_size=0.15):
    iris = load_iris()
    x, y = iris.data, iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    y_train[y_train != current_class] = 0
    y_train[y_train == current_class] = 1
    y_test[y_test != current_class] = 0
    y_test[y_test == current_class] = 1
    return x_train, x_test, y_train, y_test


def load_digits_dataset(test_size=0.15):
    digits = load_digits()
    x, y = digits.data, digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    y_train[y_train != 0] = 1
    y_test[y_test != 0] = 1
    return x_train, x_test, y_train, y_test


def lms_evaluation(x_train, x_test, y_train, y_test, eta=0.0001, max_iterations=1000000, mse_no_change=10):
    iris_lms = LMS(eta=eta, max_iterations=max_iterations, mse_no_change=mse_no_change)
    lms, lms_iterations, lms_mse = iris_lms.fit(x_train, y_train)
    print(f"LMS number of iterations to coverage: {lms_iterations}")
    print(f"LMS number coverage MSE: {lms_mse}")
    y_pred = iris_lms.predict(x_test)
    print(f"LMS accuracy score: {accuracy_score(y_pred=y_pred, y_true=y_test)}")


def main():
    digits_x_train, digits_x_test, digits_y_train, digits_y_test = load_digits_dataset()

    print("Iris Dataset Compare")
    for curr_class in range(3):
        iris_x_train, iris_x_test, iris_y_train, iris_y_test = load_iris_dataset(curr_class)
        lms_evaluation(iris_x_train, iris_x_test, iris_y_train, iris_y_test)


if __name__ == '__main__':
    main()
