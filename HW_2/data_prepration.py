import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from HW_2.data_configurations import *
from collections import defaultdict
from sklearn import metrics
from impyute import imputation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    df = pd.read_csv(filepath, header=0)
    return df


def split_database(df):
    validation_after_split_size = validation_size / (1 - test_size)
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != label], df[label],
                                                        test_size=test_size,
                                                        shuffle=False, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=validation_after_split_size,
                                                      shuffle=False, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test


def categorize_data(df):
    object_columns = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    for curr_column in object_columns:
        df[curr_column] = df[curr_column].astype("category")
        df[curr_column] = df[curr_column].cat.rename_categories(range(df[curr_column].dropna().nunique()))
        df.loc[df[curr_column].isna(), curr_column] = np.nan  # fix NaN conversion
    return df


def categorize_to_float(df):
    for curr_column in object_features:
        if curr_column == label:
            continue
        df[curr_column] = df[curr_column].astype("float64")
    return df


def negative_2_NaN(x_train, x_val, x_test):
    for feature in numerical_features:
        x_train.loc[(~x_train[feature].isnull()) & (x_train[feature] < 0), feature] = np.nan
        x_val.loc[(~x_val[feature].isnull()) & (x_val[feature] < 0), feature] = np.nan
        x_test.loc[(~x_test[feature].isnull()) & (x_test[feature] < 0), feature] = np.nan
    return x_train, x_val, x_test


def export_to_csv(filespath, x_train, x_val, x_test, y_train, y_val, y_test, prefix):
    x_train = x_train.assign(Vote=y_train.values)
    x_val = x_val.assign(Vote=y_val.values)
    x_test = x_test.assign(Vote=y_test.values)
    x_train.to_csv(filespath + "{}_train.csv".format(prefix), index=False)
    x_val.to_csv(filespath + "{}_val.csv".format(prefix), index=False)
    x_test.to_csv(filespath + "{}_test.csv".format(prefix), index=False)


def remove_outliers(x_train, x_val, x_test):
    mean_train = x_train[normal_features].mean()
    std_train = x_train[normal_features].std()

    dist_train = (x_train[normal_features] - mean_train) / std_train
    dist_val = (x_val[normal_features] - mean_train) / std_train
    dist_test = (x_test[normal_features] - mean_train) / std_train

    data_list = [x_train, x_val, x_test]
    dist_list = [dist_train, dist_val, dist_test]

    for feature in normal_features:
        for df, dist in zip(data_list, dist_list):
            for i in dist[feature].loc[(dist[feature] > z_threshold) | (dist[feature] < -z_threshold)].index:
                df.at[i, feature] = np.nan

    return x_train, x_val, x_test


def get_features_correlation(data):
    correlation_dict = defaultdict(list)
    for f1 in data.columns:
        for f2 in data.columns:
            if f2 == f1:
                continue
            correlation = data[f1].corr(data[f2], method='pearson')  # calculating pearson correlation
            if abs(correlation) >= features_correlation_threshold:
                correlation_dict[f1].append(f2)
    return correlation_dict


def fill_feature_correlation(train, val, test, correlation_dict):
    for f1 in correlation_dict.keys():
        for f2 in correlation_dict[f1]:
            coef_val = (train[f2] / train[f1]).mean()

            # fill values for train set
            other_approximation = train[f1] * coef_val
            train[f2].fillna(other_approximation, inplace=True)

            # fill values for validation set
            other_approximation = val[f1] * coef_val
            val[f2].fillna(other_approximation, inplace=True)

            # fill values for test set
            other_approximation = test[f1] * coef_val
            test[f2].fillna(other_approximation, inplace=True)


def get_feature_mi(data):
    correlation_dict = defaultdict(list)
    for feature in data.columns:
        for other in data.columns:
            if other == feature:
                continue
            mi = metrics.mutual_info_score(data[feature].values, data[other].values)
            if mi >= mi_imputation_threshold:
                correlation_dict[feature].append(other)
    return correlation_dict


def distance_num(a, b, r):
    return np.divide(np.abs(np.subtract(a, b)), r)


def closest_fit(ref_data, examine_row):
    obj_features = [f for f in object_features if f in ref_data.columns]
    data_obj = ref_data[obj_features]
    example_obj = examine_row[obj_features].values
    obj_diff = data_obj.apply(lambda row: (row.values != example_obj).sum(), axis=1)

    num_features = [f for f in numerical_features if f in ref_data.columns]
    data_num = ref_data[num_features]
    example_num = examine_row[num_features]
    col_max = data_num.max().values
    col_min = data_num.min().values
    r = col_max - col_min

    data_num = data_num.replace(np.nan, np.inf)
    example_num = example_num.replace(np.nan, np.inf)

    num_diff = data_num.apply(lambda row: distance_num(row.values, example_num.values, r), axis=1)
    for row in num_diff:
        row[(row == np.inf) | np.isnan(row)] = 1

    num_diff = num_diff.apply(lambda row: row.sum())

    total_dist = num_diff + obj_diff
    examine_row.fillna(ref_data.iloc[total_dist.idxmin()], inplace=True)


def closest_fit_imputation(train_data, data_to_fill, is_train=False):
    for index, row in data_to_fill[data_to_fill.isnull().any(axis=1)].iterrows():
        if is_train:
            closest_fit(train_data.drop(index), row)
        else:
            closest_fit(train_data, row)


def imputations(x_train, x_val, x_test, y_train, y_val, y_test):
    train = x_train.assign(Vote=y_train.values)
    val = x_val.assign(Vote=y_val.values)
    test = x_test.assign(Vote=y_test.values)

    # fill missing values by using information from correlated features
    correlation_dict = get_features_correlation(train)
    fill_feature_correlation(train, val, test, correlation_dict)

    # fill missing data using closest fit
    # closest_fit_imputation(train, train, True)
    # closest_fit_imputation(train, val)
    # closest_fit_imputation(train, test)

    # fill normal distributed features using EM algorithm
    train_after_em = imputation.cs.em(np.array(train[normal_features]), loops=50, dtype='cont')
    train.loc[:, normal_features] = train_after_em

    # fill using statistics
    train.fillna(train.median(), inplace=True)
    val.fillna(train.median(), inplace=True)
    test.fillna(train.median(), inplace=True)

    train = train.drop('Vote', axis=1)
    val = val.drop('Vote', axis=1)
    test = test.drop('Vote', axis=1)

    return train, val, test


def normalization(x_train, x_val, x_test):
    scaler = StandardScaler()
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    x_train[uniform_features] = scaler2.fit_transform(x_train[uniform_features])
    x_val[uniform_features] = scaler2.transform(x_val[uniform_features])
    x_test[uniform_features] = scaler2.transform(x_test[uniform_features])

    non_uniform = [f for f in features_without_label if f not in uniform_features]

    x_train[non_uniform] = scaler.fit_transform(x_train[non_uniform])
    x_val[non_uniform] = scaler.transform(x_val[non_uniform])
    x_test[non_uniform] = scaler.transform(x_test[non_uniform])
    return x_train, x_val, x_test


def main():
    df = load_data(DATA_PATH)
    # categorized nominal attributes to int
    df = categorize_data(df)

    # export raw data to csv files
    x_train, x_val, x_test, y_train, y_val, y_test = split_database(df)
    export_to_csv(PATH, x_train, x_val, x_test, y_train, y_val, y_test, prefix="raw")

    # data cleansing
    x_train, x_val, x_test = negative_2_NaN(x_train, x_val, x_test)
    x_train, x_val, x_test = remove_outliers(x_train, x_val, x_test)

    # imputation
    imputations(x_train, x_val, x_test, y_train, y_val, y_test)

    x_train = categorize_to_float(x_train)
    x_val = categorize_to_float(x_val)
    x_test = categorize_to_float(x_test)

    # scaling
    x_train, x_val, x_test = normalization(x_train, x_val, x_test)


if __name__ == '__main__':
    main()
