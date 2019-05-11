from sklearn.ensemble import RandomForestClassifier
from data_infrastructure import *
from pandas import concat, options
import numpy as np


class ManipulateData(object):
    def __init__(self):
        self.y_test: DataFrame = None
        self.y_train: DataFrame = None
        self.x_test: DataFrame = None
        self.x_train: DataFrame = None
        self.classifier_title = "RandomForest"
        self.classifier = RandomForestClassifier(random_state=0, criterion='entropy',
                                                 min_samples_split=5, min_samples_leaf=3,
                                                 n_estimators=50)

        self._current_manipulation = None

        self._manipulations_dict = \
            {
                "Avg_education_importance_decrease": self.avg_education_importance_decrease,
                "Avg_environmental_importance_increase": self.avg_environmental_importance_increase,
                "Avg_Residancy_Altitude_increase": self.avg_residancy_altitude_increase,
            }

    def avg_education_importance_decrease(self):
        options.mode.chained_assignment = None
        self.x_test.loc[self.x_test["Avg_education_importance"] >= -0.5, "Avg_education_importance"] = np.random.uniform(-1, -0.8)

    def avg_environmental_importance_increase(self):
        options.mode.chained_assignment = None
        self.x_test.loc[self.x_test["Avg_environmental_importance"] <= 0.5, "Avg_environmental_importance"] = np.random.uniform(0.85, 1)

    def avg_residancy_altitude_increase(self):
        options.mode.chained_assignment = None
        self.x_test.loc[self.x_test["Avg_Residancy_Altitude"] <= 0.5, "Avg_Residancy_Altitude"] = np.random.uniform(0.5, 0.7)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.classifier.fit(x_train, y_train)

    def get_available_manipulations(self):
        return list(self._manipulations_dict.keys())

    def set_manipulation(self, manipulation: str):
        if manipulation not in self.get_available_manipulations():
            raise Exception("manipulation not available")
        self._current_manipulation = manipulation

    def predict(self, x_test, y_test):
        self.y_test = y_test
        self.x_test = x_test
        if self._current_manipulation is None:
            raise Exception("set manipulation first")
        self._manipulations_dict[self._current_manipulation]()
        winner_color(self.classifier, x_test)


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

    manipulator = ManipulateData()
    manipulator.fit(concat([x_train, x_val], axis=0, join='outer', ignore_index=True),
                    concat([y_train, y_val], axis=0, join='outer', ignore_index=True))

    for man in manipulator.get_available_manipulations():
        manipulator.set_manipulation(man)
        manipulator.predict(x_test, y_test)


if __name__ == '__main__':
    main()
