import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class ClassifiersWrapped(object):
    def __init__(self):
        self.clf1 = RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=4, min_samples_leaf=1, n_estimators=450)
        self.clf2 = MLPClassifier(
            hidden_layer_sizes=(150, 10),
            activation='relu',
            solver='lbfgs',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=1000,
            shuffle=True,
            random_state=0,
            tol=0.0001,
            verbose=False,
            warm_start=True,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=True,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10)
        self.clf3 = SVC(C=150, kernel='poly', degree=3, random_state=0)

    def fit(self, X, y):
        self.clf1.fit(X, y)
        self.clf2.fit(X, y)
        self.clf3.fit(X, y)

    def predict(self, X):
        y_1 = self.clf1.predict(X)
        y_2 = self.clf2.predict(X)
        y_3 = self.clf3.predict(X)

        y_pred = []

        for i in range(len(y_1)):
            if y_1[i] != y_2[i] == y_3[i]:
                y_pred.append(y_2[i])
                continue
            y_pred.append(y_1[i])

        return np.asarray(y_pred)
