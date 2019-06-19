from os import path
from pandas import DataFrame, read_csv, concat
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import operator
from collections import OrderedDict

PATH = path.dirname(path.realpath(__file__)) + "/"

TRAINING_SET_PATH = PATH + "ElectionsData.csv"
TEST_SET_PATH = PATH + "ElectionsData_Pred_Features.csv"

TRAIN_PATH = PATH + "fixed_train.csv"
VALIDATION_PATH = PATH + "fixed_val.csv"
TEST_PATH = PATH + "fixed_test.csv"
EXPORT_TEST_PREDICTIONS = PATH + "test_predictions.csv"

# constants
global_train_size = 0.80
global_validation_size = 0.2
assert global_train_size + global_validation_size == 1
global_z_threshold = 4.5
global_correlation_threshold = 0.9
label = 'Vote'
global_party_in_coalition_threshold = 0.90
voters_id = "IdentityCard_Num"
# lists

selected_features = ['Vote', 'Most_Important_Issue', 'Avg_government_satisfaction',
                     'Avg_education_importance', 'Avg_environmental_importance',
                     'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
                     'Avg_monthly_expense_on_pets_or_plants', 'Weighted_education_rank',
                     'Number_of_valued_Kneset_members']

selected_features_without_label = ['Avg_Residancy_Altitude', 'Avg_education_importance', 'Avg_environmental_importance',
                                   'Avg_government_satisfaction', 'Avg_monthly_expense_on_pets_or_plants', 'Most_Important_Issue',
                                   'Number_of_valued_Kneset_members', 'Weighted_education_rank', 'Yearly_ExpensesK']

selected_nominal_features = ['Most_Important_Issue']

selected_numerical_features = ['Avg_Residancy_Altitude', 'Avg_education_importance', 'Avg_environmental_importance',
                               'Avg_government_satisfaction', 'Avg_monthly_expense_on_pets_or_plants', 'Number_of_valued_Kneset_members',
                               'Weighted_education_rank', 'Yearly_ExpensesK']

selected_uniform_features = ['Avg_government_satisfaction', 'Avg_education_importance',
                             'Avg_environmental_importance',
                             'Avg_Residancy_Altitude', 'Yearly_ExpensesK']

selected_normal_features = ['Avg_monthly_expense_on_pets_or_plants',
                            'Number_of_valued_Kneset_members']

label2num = {'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4, 'Oranges': 5, 'Pinks': 6, 'Purples': 7, 'Reds': 8,
             'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}

num2label = {0: 'Blues', 1: 'Browns', 2: 'Greens', 3: 'Greys', 4: 'Khakis', 5: 'Oranges', 6: 'Pinks', 7: 'Purples', 8: 'Reds',
             9: 'Turquoises', 10: 'Violets', 11: 'Whites', 12: 'Yellows'}


def import_from_csv(filepath: str) -> DataFrame:
    df = read_csv(filepath, header=0)
    return df


def export_to_csv(filepath: str, df: DataFrame):
    df.to_csv(filepath, index=False)


def filter_possible_coalitions(possible_coalitions: dict):
    """
    :param possible_coalitions: all possible coalition
    :return: possible coalition without duplication
    """
    # remove duplicates
    filtered_possible_coalitions = dict()
    for _coalition_name, _coalition_list in possible_coalitions.items():
        _coalition_list.sort()
        if _coalition_list not in filtered_possible_coalitions.values():
            filtered_possible_coalitions[_coalition_name] = _coalition_list
    return filtered_possible_coalitions


def to_binary_class(data, value):
    """
    :param data: regular data
    :param value: the value to be assigned as 1
    :return: binary classified data
    """
    binary_data = data.copy()
    bool_labels = binary_data[label] == value
    binary_data[label] = bool_labels
    return binary_data


def divide_data(df: DataFrame, data_class=label):
    x_df = df.loc[:, df.columns != data_class]
    y_df = df[data_class]
    return x_df, y_df


def categorize_data(df: DataFrame):
    object_columns = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    for curr_column in object_columns:
        df[curr_column] = df[curr_column].astype("category")
        df[curr_column + '_Int'] = df[curr_column].cat.rename_categories(range(df[curr_column].nunique())).astype('int')
        df.loc[df[curr_column].isna(), curr_column + '_Int'] = np.nan  # fix NaN conversion
        df[curr_column] = df[curr_column + '_Int']
        df = df.drop(curr_column + '_Int', axis=1)
    return df


def split_training_set(df: DataFrame, validation_size: float) -> (
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame):
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size)
    x = df.loc[:, df.columns != label]
    y = df[label]
    train_index_first, test_index = next(val_split.split(x, y))
    x_train, x_val, y_train, y_val = x.iloc[train_index_first], x.iloc[test_index], y[train_index_first], y[test_index]

    train = x_train.assign(Vote=y_train.values).reset_index(drop=True)
    val = x_val.assign(Vote=y_val.values).reset_index(drop=True)

    x = train.loc[:, df.columns != label]
    y = train[label]

    return train, val


def score(x_train: DataFrame, y_train: DataFrame, clf, k: int):
    return cross_val_score(clf, x_train, y_train, cv=k, scoring='accuracy').mean()


def get_sorted_vote_division(y):
    vote_results = dict()
    for label_name, label_index in label2num.items():
        percent_of_voters = sum(list(y == label_index)) / len(y)
        vote_results[label_index] = percent_of_voters
    return OrderedDict(sorted(vote_results.items(), key=operator.itemgetter(1)))
