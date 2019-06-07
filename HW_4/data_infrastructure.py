from os import path
from pandas import DataFrame, read_csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import operator
from collections import OrderedDict

PATH = path.dirname(path.realpath(__file__)) + "/"
DATA_PATH = PATH + "ElectionsData.csv"
TRAIN_PATH = PATH + "fixed_train.csv"
VALIDATION_PATH = PATH + "fixed_val.csv"
TEST_PATH = PATH + "fixed_test.csv"

# constants
global_train_size = 0.65
global_test_size = 0.25
global_validation_size = 0.1
assert global_train_size + global_test_size + global_validation_size == 1
global_z_threshold = 4.5
global_correlation_threshold = 0.9
global_transportation_threshold = 0.7
label = 'Vote'
global_party_in_coalition_threshold = 0.7

# lists
selected_features = ['Vote', 'Most_Important_Issue', 'Avg_government_satisfaction',
                     'Avg_education_importance', 'Avg_environmental_importance',
                     'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
                     'Avg_monthly_expense_on_pets_or_plants', 'Weighted_education_rank',
                     'Number_of_valued_Kneset_members']

selected_features_without_label = ['Most_Important_Issue', 'Avg_government_satisfaction',
                                   'Avg_education_importance', 'Avg_environmental_importance',
                                   'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
                                   'Avg_monthly_expense_on_pets_or_plants', 'Weighted_education_rank',
                                   'Number_of_valued_Kneset_members']

selected_nominal_features = ['Most_Important_Issue']

selected_numerical_features = ['Avg_government_satisfaction',
                               'Avg_education_importance', 'Avg_environmental_importance',
                               'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
                               'Avg_monthly_expense_on_pets_or_plants', 'Weighted_education_rank',
                               'Number_of_valued_Kneset_members']

selected_uniform_features = ['Avg_government_satisfaction', 'Avg_education_importance',
                             'Avg_environmental_importance',
                             'Avg_Residancy_Altitude', 'Yearly_ExpensesK']

selected_normal_features = ['Avg_monthly_expense_on_pets_or_plants',
                            'Number_of_valued_Kneset_members']

label2num = {'Blues': 0, 'Browns': 1, 'Greens': 2, 'Greys': 3, 'Khakis': 4, 'Oranges': 5, 'Pinks': 6, 'Purples': 7, 'Reds': 8,
             'Turquoises': 9, 'Violets': 10, 'Whites': 11, 'Yellows': 12}

num2label = {0: 'Blues', 1: 'Browns', 2: 'Greens', 3: 'Greys', 4: 'Khakis', 5: 'Oranges', 6: 'Pinks', 7: 'Purples', 8: 'Reds',
             9: 'Turquoises', 10: 'Violets', 11: 'Whites', 12: 'Yellows'}


def load_data(filepath: str) -> DataFrame:
    df = read_csv(filepath, header=0)
    return df


def load_prepared_dataFrames():
    return load_data(TRAIN_PATH), load_data(VALIDATION_PATH), load_data(TEST_PATH)


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


def get_sorted_vote_results(df):
    vote_results = dict()
    for label_name, label_index in label2num.items():
        precent_of_voters = sum(list(df.loc[:, label] == label_index)) / len(df)
        vote_results[label_index] = precent_of_voters
    return OrderedDict(sorted(vote_results.items(), key=operator.itemgetter(1)))


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


def split_database(df: DataFrame, test_size: float, validation_size: float) -> (
        DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame):
    validation_after_split_size = validation_size / (1 - test_size)
    first_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    x = df.loc[:, df.columns != label]
    y = df[label]
    train_index_first, test_index = next(first_split.split(x, y))
    x_train, x_test, y_train, y_test = x.iloc[train_index_first], x.iloc[test_index], y[train_index_first], y[test_index]

    test = x_test.assign(Vote=y_test.values).reset_index(drop=True)
    train = x_train.assign(Vote=y_train.values).reset_index(drop=True)

    x = train.loc[:, df.columns != label]
    y = train[label]

    second_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_after_split_size)
    train_index_second, val_index = next(second_split.split(x, y))
    x_train, x_val, y_train, y_val = x.iloc[train_index_second], x.iloc[val_index], y[train_index_second], y[val_index]

    train = x_train.assign(Vote=y_train.values).reset_index(drop=True)
    val = x_val.assign(Vote=y_val.values).reset_index(drop=True)

    return train, val, test


def export_to_csv(filespath: str, x_train: DataFrame, x_val: DataFrame,
                  x_test: DataFrame, y_train: DataFrame, y_val: DataFrame,
                  y_test: DataFrame, prefix: str):
    x_train = x_train.assign(Vote=y_train.values)
    x_val = x_val.assign(Vote=y_val.values)
    x_test = x_test.assign(Vote=y_test.values)
    x_train.to_csv(filespath + "{}_train.csv".format(prefix), index=False)
    x_val.to_csv(filespath + "{}_val.csv".format(prefix), index=False)
    x_test.to_csv(filespath + "{}_test.csv".format(prefix), index=False)


def score(x_train: DataFrame, y_train: DataFrame, clf, k: int):
    return cross_val_score(clf, x_train, y_train, cv=k, scoring='accuracy').mean()


def __distance_num(a, b, r):
    np.seterr(invalid='ignore')
    return np.divide(np.abs(np.subtract(a, b)), r)


def closest_fit(ref_data, examine_row, local_nominal_features,
                local_numerical_features):
    current_nominal_features = [f for f in local_nominal_features if
                                f in ref_data.columns]
    data_nominal = ref_data[current_nominal_features]
    examine_row_obj = examine_row[current_nominal_features].values
    obj_diff = data_nominal.apply(
        lambda _row: (_row.values != examine_row_obj).sum(), axis=1)

    num_features = [f for f in local_numerical_features if
                    f in ref_data.columns]
    data_numerical = ref_data[num_features]
    examine_row_numerical = examine_row[num_features]
    col_max = data_numerical.max().values
    col_min = data_numerical.min().values
    r = col_max - col_min

    # replace missing values in examine row to inf in order distance to work
    examine_row_numerical = examine_row_numerical.replace(np.nan, np.inf)

    num_diff = data_numerical.apply(
        lambda _row: __distance_num(_row.values, examine_row_numerical.values,
                                    r), axis=1)
    for row in num_diff:
        row[(row == np.inf)] = 1

    num_diff = num_diff.apply(lambda _row: _row.sum())

    total_dist = num_diff + obj_diff
    return total_dist.reset_index(drop=True).idxmin()


def export_to_csv(filepath: str, df: DataFrame):
    df.to_csv(filepath, index=False)
