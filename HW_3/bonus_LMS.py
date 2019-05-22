from data_infrastructure import *
from sklearn.datasets import load_iris, load_digits, make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt


class AdalineSGD(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs
        self.weights = None
        self.cost = None

    def train(self, x_train, y_train, reinitialize_weights=True):

        if reinitialize_weights:
            self.weights = np.zeros(1 + x_train.shape[1])
        self.cost = []

        for i in range(self.epochs):
            for xi, target in zip(x_train, y_train):
                output = self.net_input(xi)
                error = (target - output)
                self.weights[1:] += self.eta * xi.dot(error)
                self.weights[0] += self.eta * error

            cost = ((y_train - self.activation(x_train)) ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


def one_vs_all(y, class_index):
    return np.vectorize(lambda t: 1 if t == class_index else -1)(y.astype('int').copy())


def load_iris_dataset(test_size=0.30):
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


def __lms_evaluation(current_class, x_train, x_test, y_train, y_test, lr, max_epochs):
    print(f"current class is {current_class}")
    train_error = []
    test_error = []
    for current_max in range(max_epochs):
        adaline = AdalineSGD(eta=lr, epochs=current_max)
        adaline.train(x_train, one_vs_all(y_train, current_class))
        y_train_pred = adaline.predict(x_train)
        y_test_pred = adaline.predict(x_test)
        train_error.append(1 - accuracy_score(y_true=one_vs_all(y_train, current_class), y_pred=y_train_pred))
        test_error.append(1 - accuracy_score(y_true=one_vs_all(y_test, current_class), y_pred=y_test_pred))

    plt.plot(train_error, 'b', label="Train")
    plt.plot(test_error, 'r', label="Test")
    plt.xlabel("# Epochs")
    plt.ylabel("Error rate")
    plt.legend()
    min_error = min(test_error)
    epoch = test_error.index(min_error)
    plt.show()
    print(f"convergence epoch is: {epoch}")
    print(f"Train: {100 - 100 * train_error[epoch]:.2f}%")
    print(f"Test: {100 - 100 * test_error[epoch]:.2f}%")


def lms_iris_and_digits():
    print("Iris Dataset LMS")
    iris_x_train, iris_x_test, iris_y_train, iris_y_test = load_iris_dataset(test_size=0.30)
    for current_class in range(3):
        __lms_evaluation(current_class, iris_x_train, iris_x_test, iris_y_train, iris_y_test, 0.001, 300)

    print("Digits Dataset LMS")
    digits_x_train, digits_x_test, digits_y_train, digits_y_test = load_digits_dataset(test_size=0.30)
    for current_class in range(10):
        __lms_evaluation(current_class, digits_x_train, digits_x_test, digits_y_train, digits_y_test, 0.001, 100)


def __perceptron_evaluation(current_class, x_train, x_test, y_train, y_test, lr, max_epochs, tol=1e-3, n_iter_no_change=5):
    print(f"current class is {current_class}")

    perceptron = Perceptron(alpha=lr, max_iter=max_epochs, verbose=True, tol=tol, n_iter_no_change=n_iter_no_change)
    perceptron.fit(x_train, one_vs_all(y_train, current_class))
    y_train_pred = perceptron.predict(x_train)
    y_test_pred = perceptron.predict(x_test)
    train_error = 1 - accuracy_score(y_true=one_vs_all(y_train, current_class), y_pred=y_train_pred)
    test_error = 1 - accuracy_score(y_true=one_vs_all(y_test, current_class), y_pred=y_test_pred)

    print(f"Train: {100 - 100 * train_error:.2f}%")
    print(f"Test: {100 - 100 * test_error:.2f}%")
    return test_error


def perceptron_iris_and_digits():
    print("Iris Dataset Perceptron")
    iris_x_train, iris_x_test, iris_y_train, iris_y_test = load_iris_dataset(test_size=0.30)
    for current_class in range(3):
        __perceptron_evaluation(current_class, iris_x_train, iris_x_test, iris_y_train, iris_y_test, 0.0001, 300)

    print("Digits Dataset Perceptron")
    digits_x_train, digits_x_test, digits_y_train, digits_y_test = load_digits_dataset(test_size=0.30)
    for current_class in range(10):
        __perceptron_evaluation(current_class, digits_x_train, digits_x_test, digits_y_train, digits_y_test, 0.0001, 300)


def loading_dataset_for_lms(test_size=0.30):
    separable = True
    while separable:
        samples = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                      n_informative=1, n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    plt.plot(red[:, 0], red[:, 1], 'r.')
    plt.plot(blue[:, 0], blue[:, 1], 'b.')
    plt.show()

    sss = StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=test_size)
    x, y = samples

    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


def loading_dataset_for_perceptron(test_size=0.30):
    separable = False
    while not separable:
        samples = make_classification(random_state=0, n_samples=100, n_features=2, n_redundant=0,
                                      n_informative=1, n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    plt.plot(red[:, 0], red[:, 1], 'r.')
    plt.plot(blue[:, 0], blue[:, 1], 'b.')
    plt.show()

    sss = StratifiedShuffleSplit(random_state=0, n_splits=1, test_size=test_size)
    x, y = samples

    train_index_first, test_index = next(sss.split(x, y))
    x_train, x_test, y_train, y_test = x[train_index_first], x[test_index], y[train_index_first], y[test_index]
    return x_train, x_test, y_train, y_test


def lms_convergent_perceptron_no():
    x_train, x_test, y_train, y_test = loading_dataset_for_lms()
    __lms_evaluation(0, x_train, x_test, y_train, y_test, lr=0.001, max_epochs=10)
    test_error = []
    for current_max in range(1, 101):
        test_error.append(__perceptron_evaluation(0, x_train, x_test, y_train, y_test, lr=0.001, max_epochs=current_max, tol=None))
    plt.plot(test_error, 'r', label="Test")
    plt.xlabel("# Epochs")
    plt.ylabel("Error rate")
    min_error = min(test_error)
    epoch = test_error.index(min_error)
    plt.show()
    print(f"Perceptron best epoch is {epoch}")


def perceptron_convergent_lms_no():
    x_train, x_test, y_train, y_test = loading_dataset_for_perceptron()
    __lms_evaluation(0, x_train, x_test, y_train, y_test, lr=0.001, max_epochs=10)
    test_error = []
    for current_max in range(1, 101):
        test_error.append(__perceptron_evaluation(0, x_train, x_test, y_train, y_test, lr=0.001, max_epochs=current_max, tol=None))
    plt.plot(test_error, 'r', label="Test")
    plt.xlabel("# Epochs")
    plt.ylabel("Error rate")
    min_error = min(test_error)
    epoch = test_error.index(min_error)
    plt.show()
    print(f"Perceptron best epoch is {epoch}")


def main():
    np.random.seed(0)
    lms_iris_and_digits()
    perceptron_iris_and_digits()
    lms_convergent_perceptron_no()
    perceptron_convergent_lms_no()


if __name__ == '__main__':
    main()
