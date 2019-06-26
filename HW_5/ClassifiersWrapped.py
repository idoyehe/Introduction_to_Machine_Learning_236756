from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from data_infrastructure import *
from joblib import dump, load


class ClassifiersWrapped(object):
    def __init__(self, k, n_iter_search):
        self.n_iter_search = n_iter_search
        self.k = k
        self.clf1 = RandomForestClassifier()
        self.clf2 = MLPClassifier()
        self.clf3 = SVC()

        self.dump_clf1 = "random_forest.joblib"
        self.dump_clf2 = "mlp.joblib"
        self.dump_clf3 = "svc.joblib"

    def fit(self, X, y):
        self.clf1.fit(X, y)
        self.clf2.fit(X, y)
        self.clf3.fit(X, y)

    def classifier1_k_cross_valdition(self, X, y):
        param_dist = {"max_depth": [None],
                      "random_state": [0],
                      "max_features": sp_randint(1, len(selected_features_without_label)),
                      "min_samples_split": sp_randint(2, 7),
                      "min_samples_leaf": sp_randint(1, 7),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"],
                      "n_estimators": sp_randint(350, 450)}
        self.clf1 = RandomizedSearchCV(self.clf1, param_distributions=param_dist, refit=True,
                                       n_iter=self.n_iter_search, cv=self.k, iid=False).fit(X, y)
        clf1_best_score = self.clf1.best_score_
        self.clf1 = self.clf1.best_estimator_
        dump(self.clf1, PATH + self.dump_clf1)
        return clf1_best_score

    def classifier2_k_cross_valdition(self, X, y):
        class Layers(object):
            def rvs(self, random_state):
                return sp_randint(80, 200).rvs(), sp_randint(5, 12).rvs()

        param_dist = {"hidden_layer_sizes": Layers(),
                      "random_state": [0],
                      "activation": ['relu'],
                      "solver": ['lbfgs'],
                      "alpha": [0.01, 0.001, 0.0001],
                      "learning_rate": ['adaptive', 'constant'],
                      "max_iter": sp_randint(1000, 1800),
                      "shuffle": [True]}

        self.clf2 = RandomizedSearchCV(self.clf2, param_distributions=param_dist, refit=True,
                                       n_iter=self.n_iter_search, cv=self.k, iid=False).fit(X, y)
        clf2_best_score = self.clf2.best_score_
        self.clf2 = self.clf2.best_estimator_
        dump(self.clf1, PATH + self.dump_clf2)
        return clf2_best_score

    def classifier3_k_cross_valdition(self, X, y):
        param_dist = {"C": [10, 50, 100, 150, 1000],
                      "kernel": ['rbf', 'poly'],
                      "degree": [2, 3, 4],
                      "random_state": [0]}

        self.clf3 = RandomizedSearchCV(self.clf3, param_distributions=param_dist, refit=True,
                                       n_iter=self.n_iter_search, cv=self.k, iid=False).fit(X, y)

        clf3_best_score = self.clf3.best_score_
        self.clf3 = self.clf3.best_estimator_
        dump(self.clf1, PATH + self.dump_clf3)
        return clf3_best_score

    def clf1_predict(self, X):
        return self.clf1.predict(X)

    def clf2_predict(self, X):
        return self.clf2.predict(X)

    def clf3_predict(self, X):
        return self.clf3.predict(X)

    def load_clfs(self):
        self.clf1 = load(self.dump_clf1)
        self.clf2 = load(self.dump_clf2)
        self.clf3 = load(self.dump_clf3)
