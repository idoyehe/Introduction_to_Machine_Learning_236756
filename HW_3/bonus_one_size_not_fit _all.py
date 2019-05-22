from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from data_infrastructure import *
from pandas import concat
import numpy as np


class ClassifierPerTask(object):
    def __init__(self):
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.classifiers_dict = {
            "RandomForest1": RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=3,
                                                    min_samples_leaf=1, n_estimators=100),
            "RandomForest2": RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=5,
                                                    min_samples_leaf=3, n_estimators=50),
            "RandomForest3": RandomForestClassifier(random_state=0, criterion='gini', min_samples_split=3,
                                                    min_samples_leaf=1, n_estimators=100),
            "SGD": SGDClassifier(random_state=0, max_iter=1000, tol=1e-3, loss='log'),
            "KNN": KNeighborsClassifier(n_neighbors=3, algorithm='auto')
        }

        self.winner_party_dict = {}
        self.division_voters_dict = {}
        self.best_precision_dict = None
        self.best_f_1_dict = None
        self.best_recall_dict = None

    def _resolve_winner_party_clf(self):
        print("resolve_winner_party_clf")
        winner_ref = int(self.y_train.value_counts(sort=True).idxmax())
        for clf_title, clf in self.classifiers_dict.items():
            fitted_clf = clf.fit(self.x_train, self.y_train)
            y_test_proba: np.ndarray = np.average(fitted_clf.predict_proba(self.x_test), axis=0)
            pred_winner = np.argmax(y_test_proba)
            self.winner_party_dict[clf_title] = pred_winner == winner_ref

    def _resolve_division_voters_clf(self):
        print("resolve_division_voters_clf")
        for clf_title, clf in self.classifiers_dict.items():
            fitted_clf = clf.fit(self.x_train, self.y_train)
            y_test_pred = fitted_clf.predict(self.x_test)
            acc_score = accuracy_score(y_true=self.y_test, y_pred=y_test_pred)
            self.division_voters_dict[clf_title] = acc_score

    def _resolve_transportation_services_clf(self):
        self.best_precision_dict = {'Blues': 0, 'Browns': 0, 'Greens': 0, 'Greys': 0, 'Khakis': 0, 'Oranges': 0, 'Pinks': 0, 'Purples': 0,
                                    'Reds': 0, 'Turquoises': 0, 'Violets': 0, 'Whites': 0, 'Yellows': 0}

        self.best_recall_dict = {'Blues': 0, 'Browns': 0, 'Greens': 0, 'Greys': 0, 'Khakis': 0, 'Oranges': 0, 'Pinks': 0, 'Purples': 0,
                                 'Reds': 0, 'Turquoises': 0, 'Violets': 0, 'Whites': 0, 'Yellows': 0}

        self.best_f_1_dict = {'Blues': 0, 'Browns': 0, 'Greens': 0, 'Greys': 0, 'Khakis': 0, 'Oranges': 0, 'Pinks': 0, 'Purples': 0,
                              'Reds': 0, 'Turquoises': 0, 'Violets': 0, 'Whites': 0, 'Yellows': 0}

        print("resolve_transportation_services_clf")
        for color_index, _label in num2label.items():

            best_precision = 0
            best_recall = 0
            best_f_1 = 0

            for clf_title, clf in self.classifiers_dict.items():
                y_train_mofied = one_vs_all(self.y_train, color_index)
                fitted_clf = clf.fit(self.x_train, y_train_mofied)
                y_predicted: np.ndarray = fitted_clf.predict(self.x_test)
                y_target_local = one_vs_all(self.y_test, color_index)
                _precision = precision_score(y_target_local, y_predicted)
                _recall = recall_score(y_target_local, y_predicted)
                _f_1 = f1_score(y_target_local, y_predicted)

                if _precision > best_precision:
                    best_precision = _precision
                    self.best_precision_dict[_label] = (clf_title, best_precision)

                if _recall > best_recall:
                    best_recall = _recall
                    self.best_recall_dict[_label] = (clf_title, best_recall)

                if _f_1 > best_f_1:
                    best_f_1 = _f_1
                    self.best_f_1_dict[_label] = (clf_title, best_f_1)

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
    clpt.fit(concat([x_train, x_val], axis=0, join='outer', ignore_index=True),
             concat([y_train, y_val], axis=0, join='outer', ignore_index=True))
    clpt.predict(x_test, y_test)
    print(f"winner party dict->\n{clpt.winner_party_dict}\n")
    print(f"division votes dict->\n{clpt.division_voters_dict}\n")
    print(f"precision dict->\n{clpt.best_precision_dict}\n")
    print(f"recall votes dict->\n{clpt.best_recall_dict}\n")
    print(f"F 1 votes dict->\n{clpt.best_f_1_dict}\n")


if __name__ == '__main__':
    main()
