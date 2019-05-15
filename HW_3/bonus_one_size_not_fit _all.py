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
        for clf_title, clf in self.classifiers_dict.items():
            print(f"Current Classifier: {clf_title}")
            fitted_clf = clf.fit(self.x_train, self.y_train)
            y_test_pred = fitted_clf.predict(self.x_test)
            acc_score = accuracy_score(y_true=self.y_test, y_pred=y_test_pred)
            self.division_voters_dict[clf_title] = acc_score

    def _resolve_transportation_services_clf(self):
        print("resolve_transportation_services_clf")
        for clf_title, clf in self.classifiers_dict.items():
            print(f"Current Classifier: {clf_title}")
            fitted_clf = clf.fit(self.x_train, self.y_train)
            y_predicted: np.ndarray = fitted_clf.predict(self.x_test)
            precision = 0
            recall = 0
            f_1 = 0
            for color_index, _label in num2label.items():
                y_target_local, y_predicted_local = on_vs_all(self.y_test, y_predicted, color_index)
                precision += precision_score(y_target_local, y_predicted_local)
                recall += recall_score(y_target_local, y_predicted_local)
                f_1 = f1_score(y_target_local, y_predicted_local)
            self.transportation_services_dict[clf_title] = (precision / len(num2label), recall / len(num2label), f_1 / len(num2label))

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
    print(clpt.winner_party_dict)
    print(clpt.division_voters_dict)
    print(clpt.transportation_services_dict)


if __name__ == '__main__':
    main()
