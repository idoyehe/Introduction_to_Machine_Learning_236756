from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from data_infrastructure import *
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


class ClassifierPerTask(object):
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.classifiers_dict = {
            "MLP": MLPClassifier(random_state=0, hidden_layer_sizes=(100, 100),
                                 batch_size=32, learning_rate='adaptive', max_iter=1000,
                                 activation='relu'),
            "RandomForest": RandomForestClassifier(random_state=0, criterion='entropy',
                                                   min_samples_split=5, min_samples_leaf=3,
                                                   n_estimators=50),

            "SGD": SGDClassifier(random_state=0, max_iter=1000, tol=1e-3, loss='log'),
            "KNN": KNeighborsClassifier(n_neighbors=3)

        }

        self.winner_party_dict = {}
        self.division_voters_dict = {}
        self.transportation_services_dict = {}

    def _resolve_winner_party_clf(self):
        print("resolve_winner_party_clf")
        winner_ref = int(self.y_train.value_counts(sort=True).idxmax())
        for clf_title, clf in self.classifiers_dict.items():
            print(f"Current Classifier: {clf_title}")
            fitted_clf = clf.fit(self.x_train, self.y_train)
            y_test_proba: np.ndarray = np.average(fitted_clf.predict_proba(self.x_test), axis=0)
            pred_winner = np.argmax(y_test_proba)
            self.winner_party_dict[clf_title] = pred_winner == winner_ref

    def _resolve_division_voters_clf(self):
        print("resolve_division_voters_clf")
        hist_ref = np.bincount(self.y_train.values.astype('int64'))
        hist_ref = hist_ref / hist_ref.sum()
        for clf_title, clf in self.classifiers_dict.items():
            print(f"Current Classifier: {clf_title}")
            fitted_clf = clf.fit(self.x_train, self.y_train)
            y_test_pred = fitted_clf.predict(self.x_test)
            hist_pred = np.bincount(y_test_pred.astype('int64'))
            hist_pred = hist_pred / hist_pred.sum()
            dist = np.sqrt(np.power(hist_ref - hist_pred, 2).sum())
            self.division_voters_dict[clf_title] = dist

    def _resolve_transportation_services_clf(self):
        print("resolve_transportation_services_clf")
        hist_ref = np.bincount(self.y_train.values.astype('int64'))
        hist_ref = hist_ref / hist_ref.sum()
        for clf_title, clf in self.classifiers_dict.items():
            print(f"Current Classifier: {clf_title}")
            fitted_clf = clf.fit(self.x_train, self.y_train)
            clf_hist = np.zeros((1, 13), dtype=float)
            y_test_proba: np.ndarray = fitted_clf.predict_proba(self.x_test)
            for voter in y_test_proba:
                for index_color, proba in enumerate(voter):
                    if proba > global_transportation_threshold:
                        clf_hist[0][index_color] += 1

            clf_hist = clf_hist / clf_hist.sum()
            dist = np.sqrt(np.power(hist_ref - clf_hist, 2).sum())
            self.transportation_services_dict[clf_title] = dist

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.x_test = x_test
        self._resolve_winner_party_clf()
        self._resolve_division_voters_clf()
        self._resolve_transportation_services_clf()


def main():
    train_df = load_data(TRAIN_PATH)
    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    validation_df = load_data(VALIDATION_PATH)
    x_val = validation_df[selected_features_without_label]
    y_val = validation_df[label]

    test_df = load_data(TEST_PATH)
    x_test = test_df[selected_features_without_label]
    y_test = test_df[label]

    clpt = ClassifierPerTask()
    clpt.fit(pd.concat([x_train, x_val], axis=0, join='outer', ignore_index=True),
             pd.concat([y_train, y_val], axis=0, join='outer', ignore_index=True))
    clpt.predict(x_test, y_test)
    print(clpt.winner_party_dict)
    print(clpt.division_voters_dict)
    print(clpt.transportation_services_dict)


if __name__ == '__main__':
    main()
