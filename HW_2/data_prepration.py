from HW_2.features_selection import *
from HW_2.bonus_sfs import run_sfs_base_clfs
import pandas as pd
from sklearn.model_selection import train_test_split
from impyute import imputation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from csv import writer
from collections import defaultdict


def load_data(filepath: str) -> DataFrame:
    df = pd.read_csv(filepath, header=0)
    return df


def split_database(df: DataFrame, test_size: float, validation_size: float):
    validation_after_split_size = validation_size / (1 - test_size)
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != label], df[label],
                                                        test_size=test_size,
                                                        shuffle=True, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=validation_after_split_size,
                                                      shuffle=True, random_state=0)
    return x_train, x_val, x_test, y_train, y_val, y_test


def negative_2_nan(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    for feature in numerical_features:
        x_train.loc[(~x_train[feature].isnull()) & (x_train[feature] < 0), feature] = np.nan
        x_val.loc[(~x_val[feature].isnull()) & (x_val[feature] < 0), feature] = np.nan
        x_test.loc[(~x_test[feature].isnull()) & (x_test[feature] < 0), feature] = np.nan
    return x_train, x_val, x_test


def export_to_csv(filespath: str, x_train: DataFrame, x_val: DataFrame, x_test: DataFrame, y_train: DataFrame, y_val: DataFrame,
                  y_test: DataFrame, prefix: str):
    x_train = x_train.assign(Vote=y_train.values)
    x_val = x_val.assign(Vote=y_val.values)
    x_test = x_test.assign(Vote=y_test.values)
    x_train.to_csv(filespath + "{}_train.csv".format(prefix), index=False)
    x_val.to_csv(filespath + "{}_val.csv".format(prefix), index=False)
    x_test.to_csv(filespath + "{}_test.csv".format(prefix), index=False)


def export_selected_features(filename: str, selected_features_list: list):
    with open(filename, 'w') as csv_file:
        wr = writer(csv_file)
        wr.writerow(selected_features_list)


def remove_outliers(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame, z_threshold: float):
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


def get_features_correlation(data: DataFrame, features_correlation_threshold: float):
    correlation_dict = defaultdict(list)
    for f1 in data.columns:
        for f2 in data.columns:
            if f2 == f1:
                continue
            correlation = data[f1].corr(data[f2], method='pearson')  # calculating pearson correlation
            if abs(correlation) >= features_correlation_threshold:
                correlation_dict[f1].append(f2)
    return correlation_dict


def fill_feature_correlation(data: DataFrame, correlation_dict: dict):
    for f1 in correlation_dict.keys():
        for f2 in correlation_dict[f1]:
            coef_val = (data[f2] / data[f1]).mean()
            # fill values for data set
            other_approximation = data[f1] * coef_val
            data[f2].fillna(other_approximation, inplace=True)


def distance_num(a, b, r):
    np.seterr(invalid='ignore')
    return np.divide(np.abs(np.subtract(a, b)), r)


def closest_fit(ref_data, examine_row):
    current_nominal_features = [f for f in nominal_features if f in ref_data.columns]
    data_nominal = ref_data[current_nominal_features]
    examine_row_obj = examine_row[current_nominal_features].values
    obj_diff = data_nominal.apply(lambda _row: (_row.values != examine_row_obj).sum(), axis=1)

    num_features = [f for f in numerical_features if f in ref_data.columns]
    data_numerical = ref_data[num_features]
    examine_row_numerical = examine_row[num_features]
    col_max = data_numerical.max().values
    col_min = data_numerical.min().values
    r = col_max - col_min

    # replace missing values in examine row to inf in order distance to work
    examine_row_numerical = examine_row_numerical.replace(np.nan, np.inf)

    num_diff = data_numerical.apply(lambda _row: distance_num(_row.values, examine_row_numerical.values, r), axis=1)
    for row in num_diff:
        row[(row == np.inf)] = 1

    num_diff = num_diff.apply(lambda _row: _row.sum())

    total_dist = num_diff + obj_diff
    examine_row.fillna(ref_data.iloc[total_dist.reset_index(drop=True).idxmin()], inplace=True)


def closest_fit_imputation(ref_data: DataFrame, data_to_fill: DataFrame):
    for index, row in data_to_fill[data_to_fill.isnull().any(axis=1)].iterrows():
        closest_fit(ref_data, row)


def imputations(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame, y_train: DataFrame, y_val: DataFrame, y_test: DataFrame):
    train = x_train.assign(Vote=y_train.values)
    val = x_val.assign(Vote=y_val.values)
    test = x_test.assign(Vote=y_test.values)

    # fill missing values by using information from correlated features
    correlation_dict_train = get_features_correlation(train, global_correlation_threshold)

    fill_feature_correlation(train, correlation_dict_train)
    fill_feature_correlation(val, correlation_dict_train)
    fill_feature_correlation(test, correlation_dict_train)

    # fill missing data using closest fit
    # print("closest fit for train")
    # closest_fit_imputation(train.dropna(how='any'), train)
    # print("closest fit for validation")
    # closest_fit_imputation(val.dropna(how='any'), val)
    # print("closest fit for test")
    # closest_fit_imputation(test.dropna(how='any'), test)

    # fill normal distributed features using EM algorithm
    train_after_em = imputation.cs.em(np.array(train[normal_features]), loops=50, dtype='cont')
    train.loc[:, normal_features] = train_after_em

    # fill using statistics
    # for numerical feature filling by median
    train[numerical_features] = train[numerical_features].fillna(train[numerical_features].median(), inplace=False)
    val[numerical_features] = val[numerical_features].fillna(val[numerical_features].median(), inplace=False)
    test[numerical_features] = test[numerical_features].fillna(test[numerical_features].median(), inplace=False)

    # for categorical feature filling by majority
    train[nominal_features] = train[nominal_features].fillna(train[nominal_features].agg(lambda x: x.value_counts().index[0]),
                                                             inplace=False)
    val[nominal_features] = val[nominal_features].fillna(val[nominal_features].agg(lambda x: x.value_counts().index[0]), inplace=False)
    test[nominal_features] = test[nominal_features].fillna(test[nominal_features].agg(lambda x: x.value_counts().index[0]), inplace=False)

    train = train.drop(label, axis=1)
    val = val.drop(label, axis=1)
    test = test.drop(label, axis=1)

    return train, val, test


def normalization(x_train: DataFrame, x_val: DataFrame, x_test: DataFrame):
    scale_std = StandardScaler()
    scale_min_max = MinMaxScaler(feature_range=(-1, 1))
    local_uniform_features = [f for f in uniform_features if f not in nominal_features]
    x_train[local_uniform_features] = scale_min_max.fit_transform(x_train[local_uniform_features])
    x_val[local_uniform_features] = scale_min_max.transform(x_val[local_uniform_features])
    x_test[local_uniform_features] = scale_min_max.transform(x_test[local_uniform_features])

    local_non_uniform = [f for f in features_without_label if f not in uniform_features and f not in nominal_features]

    x_train[local_non_uniform] = scale_std.fit_transform(x_train[local_non_uniform])
    x_val[local_non_uniform] = scale_std.transform(x_val[local_non_uniform])
    x_test[local_non_uniform] = scale_std.transform(x_test[local_non_uniform])
    return x_train, x_val, x_test


def main():
    df = load_data(DATA_PATH)
    # categorized nominal attributes to int
    df = categorize_data(df)

    # export raw data to csv files
    x_train, x_val, x_test, y_train, y_val, y_test = split_database(df, global_test_size, global_validation_size)
    export_to_csv(PATH, x_train, x_val, x_test, y_train, y_val, y_test, prefix="raw")

    # data cleansing
    x_train, x_val, x_test = negative_2_nan(x_train, x_val, x_test)
    x_train, x_val, x_test = remove_outliers(x_train, x_val, x_test, global_z_threshold)

    # imputation
    x_train, x_val, x_test = imputations(x_train, x_val, x_test, y_train, y_val, y_test)

    # scaling
    x_train, x_val, x_test = normalization(x_train, x_val, x_test)

    # feature selection
    # our SFS for bonus task
    selected_features_svm, selected_features_knn = run_sfs_base_clfs(x_train, y_train, x_val, y_val, x_test, y_test)
    print("for SVM SFS selected features are: {}".format(selected_features_svm))
    print("for KNN SFS selected features are: {}".format(selected_features_knn))

    # filter method
    selected_numerical_features_by_variance = variance_filter(x_train[numerical_features], y_train, global_variance_threshold)
    selected_features_by_variance = selected_numerical_features_by_variance + nominal_features
    x_train = x_train[selected_features_by_variance]
    x_val = x_val[selected_features_by_variance]
    x_test = x_test[selected_features_by_variance]

    # wrapper method
    selected_features_by_mi = apply_mi_wrapper_filter(x_train, y_train)
    x_train = x_train[selected_features_by_mi]
    x_val = x_val[selected_features_by_mi]
    x_test = x_test[selected_features_by_mi]
    export_to_csv(PATH, x_train, x_val, x_test, y_train, y_val, y_test, prefix="fixed")

    final_selected_features = x_train.columns.values.tolist()
    export_selected_features(SELECTED_FEATURES_PATH, final_selected_features)


if __name__ == '__main__':
    main()
