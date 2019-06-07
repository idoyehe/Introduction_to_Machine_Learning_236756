import csv
import pickle
import datetime
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.tree import _tree
from feature_characteristics import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import itertools
import logging
import random
import os

# loading and preparing the data
# Train the necessary models and make the calculation that will help you determine a steady coalition
# For each party, identify its leading features

# Load the prepared training set
# Train at least one generative model and one clustering model
# Each training should be done via cross-validation

# Load the prepared test set
# Apply the trained models on the test set and check performance

# prediction tasks
# determine a steady coalition
# For each party, identify its leading features

cv_folds = 5
pred_label = 'Vote'
parties_list = ('Turquoises', 'Pinks', 'Greens', 'Blues', 'Greys', 'Browns', 'Purples', 'Yellows', 'Oranges', 'Reds',
                'Whites')

other_models = {'NB': GaussianNB()}


def set_attributes_type(df):
    features = list(df.columns.drop('Vote'))
    for f in features:
        if f in real_num_features:
            df[f] = df[f].astype('float32')
        elif f in general_categorical_features:
            temp_df = pd.get_dummies(df[f], prefix=f, dummy_na=True)
            df = df.drop([f], axis=1)
            df = pd.concat([df, temp_df], axis=1, copy=True)
        elif f in special_categorical_features:
            for key in special_categorical_features_dict[f].keys():
                df.loc[df[f] == key, f] = special_categorical_features_dict[f][key]
            df[f] = df[f].astype('float32')
        else:
            print('ERROR: {0} is not in features lists')
    return df


def compute_dist_matrix(_df):
    aux_df = _df.copy().drop(['Most_Important_Issue_nan'], axis=1)
    aux_nan_df = isNaN(aux_df.values)
    aux_nan_indexes = list(zip(np.where(aux_nan_df)[0], np.where(aux_nan_df)[1]))
    for i in aux_nan_indexes:
        max_value_of_column = np.max(aux_df.iloc[:, i[1]])
        min_value_of_column = np.min(aux_df.iloc[:, i[1]])
        aux_df.iloc[i[0], i[1]] = random.uniform(min_value_of_column, max_value_of_column)
    aux_df_values = RobustScaler().fit_transform(aux_df.values)
    dist_matrix = squareform(pdist(aux_df_values, metric='euclidean'))
    return dist_matrix


def isNaN(num):
    return num != num


def find_correlations(df, min_corr_value=0.9):
    correlations = df.corr()
    corr_dict = correlations.to_dict()
    for key_1 in tuple(corr_dict.keys()):
        corr_dict[key_1].pop(key_1)
        max_corr = np.max([abs(corr_dict[key_1][key_2]) for key_2 in corr_dict[key_1].keys()])
        if max_corr < min_corr_value:
            corr_dict.pop(key_1)
            continue
        else:
            corr_tup = tuple(corr_dict[key_1].items())
            arg_max = np.argmax([i[1] for i in corr_tup])
            corr_dict[key_1] = corr_tup[arg_max][0]
    return corr_dict


def fill_missing_data_with_correlations(df, corr_dict, train_df=None, df_type='train'):
    nan_df = isNaN(df.values)
    nan_indexes = list(zip(np.where(nan_df)[0], np.where(nan_df)[1]))
    if df_type == 'train':
        for i in nan_indexes:
            nan_feature = df.columns[i[1]]
            if nan_feature not in corr_dict.keys():
                continue
            corr_to_nan_feature = corr_dict[nan_feature]
            if isNaN(df.loc[df.index[i[0]], corr_to_nan_feature]):
                continue
            dists = list(abs(df.loc[:, corr_to_nan_feature] - df.loc[df.index[i[0]], corr_to_nan_feature]))
            dists[i[0]] = float('nan')
            close_neighbor_index = np.nanargmin(dists)
            df.iloc[i] = df.loc[df.index[close_neighbor_index], df.columns[i[1]]]
    elif df_type in ['test', 'validation']:
        frames = [df, train_df]
        combined_df = pd.concat(frames)
        for i in nan_indexes:
            nan_feature = df.columns[i[1]]
            if nan_feature not in corr_dict.keys():
                continue
            corr_to_nan_feature = corr_dict[nan_feature]
            if corr_to_nan_feature not in combined_df.columns:
                continue
            if isNaN(combined_df.loc[combined_df.index[i[0]], corr_to_nan_feature]):
                continue
            dists = list(abs(combined_df.loc[:, corr_to_nan_feature] - combined_df.loc[
                combined_df.index[i[0]], corr_to_nan_feature]))
            dists[:len(df)] = [float('nan')] * len(df)  # fill missing values only with train data
            close_neighbor_index = np.nanargmin(dists)
            df.iloc[i] = combined_df.loc[combined_df.index[close_neighbor_index], combined_df.columns[i[1]]]
            continue
    return df.copy()


def remove_outliers(df, threshold=4, epsilon=0, step=1.025):
    max_rows_remove = 50
    at_start = len(df)
    to_remove = np.full((len(df),), False)
    for attr in df.drop('Vote', axis=1).columns:
        mu = df[attr].mean()
        sigma = df[attr].std()
        z_score = ((df[attr] - mu) / sigma).abs() > threshold
        if np.count_nonzero(z_score) > max_rows_remove + epsilon:
            z_score = ((df[attr] - mu) / sigma).abs() > threshold * step
            if np.count_nonzero(z_score) > max_rows_remove + epsilon:
                continue
        to_remove = to_remove | z_score.as_matrix()
    df.drop(df.index[to_remove], inplace=True)
    return df


def remove_noise(df, positive_features):
    """
    place float(nan) instead of negative values.
    """
    for f in positive_features:
        df.loc[df[f] < 0, f] = float('nan')
    return df


def impute_nans_by_closest_fit(df, unscaled_df, df_type='train'):
    if df_type == 'train':
        aux_df = df.drop(['Vote'], axis=1).copy()
        dist_matrix = compute_dist_matrix(aux_df)
        df_imputed = df.copy()
        # map nan values
        nan_df = isNaN(df_imputed.values)
        nan_indexes = list(zip(np.where(nan_df)[0], np.where(nan_df)[1]))
        # fix dist matrix
        for i in range(len(dist_matrix)):
            dist_matrix[i, i] = float('nan')
        ######### fill missing real data #########
        for i in nan_indexes:
            min_value = np.nanmin(dist_matrix[i[0], :])
            min_index = dist_matrix[i[0], :].tolist().index(min_value)
            if isNaN(df_imputed.iloc[min_index, i[1]]):
                df_imputed.iloc[i[0], i[1]] = np.mean(df_imputed.iloc[:, i[1]])
            else:
                df_imputed.iloc[i[0], i[1]] = df_imputed.iloc[min_index, i[1]]
            continue
        one_hot_list = (
            'Most_Important_Issue_Education', 'Most_Important_Issue_Environment', 'Most_Important_Issue_Financial',
            'Most_Important_Issue_Foreign_Affairs', 'Most_Important_Issue_Healthcare',
            'Most_Important_Issue_Military', 'Most_Important_Issue_Other',
            'Most_Important_Issue_Social')
        nan_hot = tuple(df_imputed.loc[:, 'Most_Important_Issue_nan'] == 1)
        nan_hot_indexes = tuple(np.where(nan_hot)[0])
        for i in nan_hot_indexes:
            min_value = np.nanmin(dist_matrix[i, :])
            min_index = dist_matrix[i, :].tolist().index(min_value)
            if df_imputed.iloc[min_index, :]['Most_Important_Issue_nan'] == 1:
                continue
            for feature_name in one_hot_list:
                if df_imputed.iloc[min_index, :][feature_name] == 1:
                    df_imputed.loc[df_imputed.index[i], feature_name] = 1
                    break
            continue
        return df_imputed
    elif df_type in ['test', 'validation']:
        test_len = len(df.index)
        aux_df = pd.concat([df, unscaled_df]).copy()
        aux_df = aux_df.drop(['Vote'], axis=1)
        aux_nan_df = isNaN(aux_df.values)
        aux_nan_indexes = list(zip(np.where(aux_nan_df)[0], np.where(aux_nan_df)[1]))
        for i in aux_nan_indexes:
            mean_value_of_column = np.nanmean(aux_df.iloc[test_len:, i[1]])
            aux_df.iloc[i[0], i[1]] = mean_value_of_column
        scale_values = unscaled_df.drop(['Vote'], axis=1).values
        scale = RobustScaler().fit(scale_values)
        aux_df_values = scale.transform(aux_df.values)
        dist_matrix = squareform(pdist(aux_df_values, metric='euclidean'))
        for i in range(len(dist_matrix)):
            dist_matrix[i, i] = float('nan')
        df_imputed = df.copy()
        nan_df = isNaN(df_imputed.values)
        nan_indexes = list(zip(np.where(nan_df)[0], np.where(nan_df)[1]))
        for i in nan_indexes:
            min_value = np.nanmin(dist_matrix[i[0], test_len:])
            min_index = np.nanargmin(dist_matrix[i[0], test_len:]) + test_len
            df_imputed.iloc[i[0], i[1]] = aux_df.iloc[min_index, i[1] - 1]
            continue
        return df_imputed


def data_preperation(input_df, typ, positive_features, unscaled_df_train=None, unfiltered_df=None, corr_dict=None):
    """
    input_df: pandas df
    typ: 'train', 'validation' or 'test'
    noise_params: list of all pure positive features
    rdnt_atts: list of all features that are redundant acc. to past data analysis
    dist_matrix: in order to fill missing
    rmv_outliers_params:
    norm_params:
    """
    df = input_df.copy()
    if typ == 'test' or typ == 'validation':
        unscaled_df = unscaled_df_train.copy()

    """ >>  Fix attribute types << """
    df = set_attributes_type(df)

    """ >>  Remove noise << """
    df = remove_noise(df, positive_features)

    """ >>  Find correlations << """
    if typ == 'train':
        corr_dict = find_correlations(df)

    """ >>  Impute missing data by correlation findings << """
    if typ == 'train':
        df = fill_missing_data_with_correlations(df, corr_dict, 'train')
    elif typ == 'test' or typ == 'validation':
        df = fill_missing_data_with_correlations(df, corr_dict, unfiltered_df, typ)

    """ >>  Remove unnecessary attributes << """
    if typ == 'train':
        unfiltered_df = df.copy()
    for feature in df.columns:
        if feature not in features_to_keep:
            df.drop([feature], axis=1, inplace=True)

    """ >>  Impute missing data by closest fit << """
    if typ == 'train':
        df = impute_nans_by_closest_fit(df, 'train')  ### >> impute all nan values in data

    elif typ == 'test' or typ == 'validation':
        df = impute_nans_by_closest_fit(df, unscaled_df, typ)

    """ >>  Remove outliers << """
    if typ == 'train':
        df = remove_outliers(df)

    """ >> normalize the data << """
    if typ == 'train':
        unscaled_df = df.copy()
        scaler = RobustScaler()
        df[list(local_atts_to_scale)] = scaler.fit_transform(df[list(local_atts_to_scale)])

    elif typ == 'test' or typ == 'validation':
        scaler_values = unscaled_df_train[list(local_atts_to_scale)].values
        scaler = RobustScaler().fit(scaler_values)
        df[list(local_atts_to_scale)] = scaler.transform(df[list(local_atts_to_scale)])

    if typ == 'train':
        df.drop('Most_Important_Issue_nan', axis=1, inplace=True)
        return df, unscaled_df, unfiltered_df, corr_dict

    elif typ == 'test' or typ == 'validation':
        df.drop('Most_Important_Issue_nan', axis=1, inplace=True)
        return df


def generate_options(max, min, step):
    options = np.arange(min, max + step, step)
    for opt in options:
        yield opt


def generate_knn_models(max_neighbors=19, min_neighbors=5, step=2):
    for neighbors_opt in generate_options(max_neighbors, min_neighbors, step):
        model_name = "{}-nn".format(neighbors_opt)
        yield model_name, KNeighborsClassifier(neighbors_opt)


def generate_rand_forest_models(n_estimators_min=5, n_estimators_max=29, n_estimators_step=2):
    for n_estimators_opt in generate_options(n_estimators_max, n_estimators_min, n_estimators_step):
        for criterion_opt in ["gini", "entropy"]:
            model_name = '{}-estimators-{}-criterion-rand-forest'.format(n_estimators_opt, criterion_opt)
            yield model_name, RandomForestClassifier(n_estimators=n_estimators_opt, criterion=criterion_opt)


def generate_dt_models(max_depth=30, min_depth=5, step=5):
    for depth_opt in generate_options(max_depth, min_depth, step):
        model_name = "{}-depth-dt".format(depth_opt)
        yield model_name, DecisionTreeClassifier(max_depth=depth_opt)


def divide_data(data, label=pred_label):
    X = data.loc[:, data.columns != label]
    y = data[label]
    return X, y


def choose_hyper_parameter(models, X, y):
    """
    :param models: list of same-family models with different parameters
    :return: best model as tuple (name, model, score)
    """
    best_score = 0
    best_model = None
    for _name, _model in models:
        scores = cross_val_score(_model, X=X, y=y, cv=cv_folds)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = _name, _model, best_score
    return best_model


def choose_hyper_parameter_for_all(data, models_dict, label=pred_label):
    """
    :param data: usually train data
    :param models_dict: a dict with the form: {<model-family-name>: <list of models with different hyper parameters>}
                        e.g {"KNN's": [<10-nn>, <20-nn>]}
    :return: list of models with the best hyper parameters. list example: [("10-nn", <10-nn model>), ("20-nn", <20-nn model>)]
    """
    tuned = []
    X, y = divide_data(data, label)
    for family_name, models_list in models_dict.items():
        _name, _model, _score = choose_hyper_parameter(models_list, X, y)
        tuned.append((_name, _model))
        logging.info("Family: {} - Model name: {} - Score: {}".format(family_name, _name, _score))
    return tuned


def train_model(_model, X, y):
    _model.fit(X, y)
    return _model


def validate_model(_model, X, y_truth, metric=accuracy_score):
    y_pred = _model.predict(X)
    score = metric(y_truth, y_pred)
    return score


def choose_model(models, train_data, val_data, label=pred_label, metric=accuracy_score):
    """
    :param models: list of different models i.e different classifiers
    :return: best model as tuple (name, model, score)
    """
    logging.info('Running model comparison...')
    X_train, y_train = divide_data(train_data, label)
    X_val, y_val = divide_data(val_data, label)
    best_score = 0
    best_model = None
    for _name, _model in models:
        trained = train_model(_model, X_train, y_train)
        curr_score = validate_model(trained, X_val, y_val, metric)
        if curr_score > best_score:
            best_score = curr_score
            best_model = _name, _model, best_score
    logging.info("Best model - {} - Accuracy: {:.3f}".format(best_model[0], best_model[2]))
    return best_model


def test_model(_model, data, label=pred_label):
    X, _ = divide_data(data, label)
    return _model.predict(X)


def aggregate_samples(y_pred):
    """
    :param y_pred:
    :return: a dictionary with the form {<label>: <list of indices>} e.g {"pinks": [1799, 6, 56]} (the list is the
            samples associated with the label)
    """
    label_dict = {}
    for label in np.unique(y_pred):
        label_dict[label] = list(np.argwhere(y_pred == label).squeeze())
    return label_dict


def get_test_accuracy(df_test, df_train, model):
    X_test, y_test = divide_data(df_test)
    X_train, y_train = divide_data(df_train)
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_predict)
    return test_accuracy


def get_test_confusion_matrix(df_test, df_train, model):
    X_test, y_test = divide_data(df_test)
    X_train, y_train = divide_data(df_train)
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    confusion_matrix_labels = df_test['Vote'].unique()
    test_confusion_matrix = confusion_matrix(y_test, y_test_predict, confusion_matrix_labels)
    test_error = sum(y_test_predict != y_test) / len(y_test)
    return test_confusion_matrix, confusion_matrix_labels, test_error


def get_sorted_vote_results(df):
    vote_results = list()
    for party in parties_list:
        sum_of_voters = sum(list(df.loc[:, 'Vote'] == party)) / len(df)
        vote_results.append([party, sum_of_voters])
    vote_results.sort(key=lambda x: x[1])
    return vote_results


def get_major_party(df):
    sorted_vote_results = get_sorted_vote_results(df)
    return sorted_vote_results[-1][0]


def get_predicted_major_party(df, model):
    y_pred = test_model(model, df)
    vote_results = list(Counter(y_pred).items())
    vote_results.sort(key=lambda x: x[1])
    predicted_major_party = vote_results[-1][0]
    return predicted_major_party


def get_division_vote_results(votes):
    num_of_votes = float(len(votes))
    raw_vote_results = list(Counter(votes).items())
    division_vote_results = dict([[vr[0], vr[1] / num_of_votes] for vr in raw_vote_results])
    return division_vote_results


def get_quad_error_from_vote_results_dicts(dict1, dict2):
    quad_error = 0
    for party_name in parties_list:
        quad_error += (dict1[party_name] - dict2[party_name]) ** 2
    return quad_error ** 0.5


def get_q3_score(df, trained_model):
    real_voters_dict = aggregate_samples(df['Vote'])
    predicted_voters = test_model(trained_model, df)
    predicted_voters_dict = aggregate_samples(predicted_voters)
    sum_of_correct_predictions = 0
    for party in parties_list:
        s1 = set(real_voters_dict[party])
        s2 = set(predicted_voters_dict[party])
        sum_of_correct_predictions += len(set.intersection(s1, s2))
    q3_score = sum_of_correct_predictions / len(predicted_voters)
    return q3_score


def get_best_predicted_voters_dict(df_test, df_valid, tuned_list):
    """ answers q3 with best available model """
    q3_score_dict = dict()
    best_q3_model_name, best_q3_model = None, None  # [model name, model]
    best_q3_model_score = 0
    for model_name, trained_model in tuned_list:
        q3_score_dict[model_name] = get_q3_score(df_valid, trained_model)
        logging.info('{0} score is: {1:3.6f}'.format(model_name, q3_score_dict[model_name]))
        if q3_score_dict[model_name] > best_q3_model_score:
            best_q3_model_name = model_name
            best_q3_model = trained_model
            best_q3_model_score = q3_score_dict[model_name]
    logging.info('Best model is {0} with score of {1:3.6f}'.format(best_q3_model_name, best_q3_model_score))
    best_predicted_voters = test_model(best_q3_model, df_test)
    best_predicted_voters_dict = aggregate_samples(best_predicted_voters)
    return best_predicted_voters_dict


def split_data_frame(raw_df):
    logging.info('splitting data...')
    raw_df_train, raw_df_test, raw_df_valid = \
        np.split(raw_df.sample(frac=1), [int(.7 * len(raw_df)), int(.85 * len(raw_df))])
    raw_df_train.to_csv('./csv/raw/raw_train.csv', sep=',', encoding='utf-8', index=False)
    raw_df_valid.to_csv('./csv/raw/raw_valid.csv', sep=',', encoding='utf-8', index=False)
    raw_df_test.to_csv('./csv/raw/raw_test.csv', sep=',', encoding='utf-8', index=False)
    return raw_df_train, raw_df_valid, raw_df_test


def create_prepared_csv_dfs(raw_df_train, raw_df_valid, raw_df_test):
    logging.info('preparing test data...')
    df_train, unscaled_df_train, unfiltered_df, corr_dict = \
        data_preperation(raw_df_train, 'train', positive_features)
    df_train.to_csv('./csv/prepared/prepared_train.csv', sep=',', encoding='utf-8', index=False)
    logging.info('preparing validation data...')
    df_valid = data_preperation(raw_df_valid, 'validation', positive_features, unscaled_df_train, unfiltered_df,
                                corr_dict)
    df_valid.to_csv('./csv/prepared/prepared_valid.csv', sep=',', encoding='utf-8', index=False)
    logging.info('preparing test data...')
    df_test = data_preperation(raw_df_test, 'test', positive_features, unscaled_df_train, unfiltered_df, corr_dict)
    df_test.to_csv('./csv/prepared/prepared_test.csv', sep=',', encoding='utf-8', index=False)
    pass


def get_legal_coalition_sets(df_train):
    legal_coalition_sets = []
    vote_results = dict(get_sorted_vote_results(df_train))
    for num_of_parties_in_coalition in range(2, len(parties_list)):
        for possible_set in itertools.combinations(parties_list, num_of_parties_in_coalition):
            if sum([vote_results[_party] for _party in possible_set]) > 0.51:
                legal_coalition_sets.append(possible_set)
    return tuple(legal_coalition_sets)


def generate_kmeans_models():
    for num_of_clusters in [2, 3, 4, 5]:
        cluster_name = "{}_clusters".format(num_of_clusters)
        cluster = KMeans(num_of_clusters)
        yield cluster_name, cluster


def get_clusters(df_train, clusters_to_check):
    n_splits = 3
    clusters = []
    kf = KFold(n_splits)
    X, y = divide_data(df_train)
    for family_name, models_list in clusters_to_check.items():
        for model_name, cluster_model in models_list:
            _index = 0
            for train_i, test_i in kf.split(X):
                _name = "{}_{}_fold-{}".format(family_name, model_name, _index)
                _model = cluster_model.fit(X.iloc[train_i, :])
                _labels = _model.predict(X.iloc[test_i, :])
                clusters.append((_name, test_i, _labels, _model))
                _index += 1
    return clusters


def get_possible_coalitions(df, clusters):
    possible_coalitions = dict()
    for cluster_name, indexes, labels, model in clusters:
        X, y = divide_data(df.iloc[indexes, :])
        for group in set(labels):
            _group_votes = y[labels == group]
            _parties_in_group = list(set(_group_votes))
            # remove parties that are not really in group (less then 80%)
            # party is in group if more then 90% of its voters are in group.
            for _party in tuple(_parties_in_group):
                _party_votes = sum(y == _party)
                _party_voters_in_group = sum(_group_votes == _party)
                _group_voters_proportion = _party_voters_in_group / _party_votes
                if _group_voters_proportion < 0.90:
                    _parties_in_group.remove(_party)
            _vote_results = dict(get_sorted_vote_results(df))
            _group_size = sum([_vote_results[p] for p in _parties_in_group])
            if _group_size > 0.51:
                possible_coalitions['{}_{}'.format(cluster_name, group)] = _parties_in_group
    # remove duplicates
    filtered_possible_coalitions = dict()
    for _coalition_name, _coalition_list in possible_coalitions.items():
        _coalition_list.sort()
        if _coalition_list not in filtered_possible_coalitions.values():
            filtered_possible_coalitions[_coalition_name] = _coalition_list
    return filtered_possible_coalitions


def get_coalition_total_variance(df, coalition_parties):
    df_coalition = df.loc[df['Vote'].isin(coalition_parties), :]
    coalition_features = df_coalition.drop('Vote', axis=1).columns
    # coalition_total_variance = sum([np.var(df_coalition[f]) for f in coalition_features])
    coalition_total_variance = sum([np.var(df_coalition[f]) for f in coalition_features[:6]]) + \
                               sum([np.var(df_coalition[f]) for f in coalition_features[6:]]) / len(
        coalition_features[6:])
    return coalition_total_variance


def get_most_homogeneous_coalition(df, possible_coalitions):
    max_total_variance = 99999
    for _coalition_name, _coalition_parties in possible_coalitions.items():
        _coalition_total_variance = get_coalition_total_variance(df, _coalition_parties)
        if _coalition_total_variance < max_total_variance:
            max_total_variance = _coalition_total_variance
            most_homogeneous_coalition_name = _coalition_name
            most_homogeneous_coalition_parties = _coalition_parties
    return most_homogeneous_coalition_name, most_homogeneous_coalition_parties


def generate_gmm_models():
    for num_of_clusters in [2, 3, 4, 5]:
        cluster_name = "{}_clusters".format(num_of_clusters)
        cluster = GaussianMixture(n_components=num_of_clusters)
        yield cluster_name, cluster


def to_binary_label(data, value, label=pred_label):
    """
    :param data: regular data
    :param value: the value to be assigned as True
    :param label:
    :return: ask me :| it was hard to describe haha :)
    """
    binary_data = data.copy()
    bool_labels = binary_data[label] == value
    binary_data[label] = bool_labels
    return binary_data


def tree_walk_aux(tree, features, start_node=0, with_leaves=False):
    """
    :param tree: the tree object extracted from DecisionTreeModel.tree_
    :param features: the feature labels
    :param start_node:
    :return: the rules list
    """
    feature_names = [features[i] if i != _tree.TREE_UNDEFINED else 'undefined' for i in tree.feature]
    rules = []

    def tree_walk(node):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[node]
            threshold = tree.threshold[node]
            left_rule = "{} <= {:.3f}".format(name, float(threshold))
            rules.append(left_rule)
            tree_walk(tree.children_left[node])
            right_rule = "{} > {:.3f}".format(name, float(threshold))
            rules.append(right_rule)
            tree_walk(tree.children_right[node])
        else:
            if with_leaves:
                rules.append("leaf")

    tree_walk(start_node)
    return rules


def get_dt_split_rules(data, max_features=5, criterion='gini'):
    """
    :param data: binary label data which means the pred_label is True/False
    :param max_features:
    :param criterion:
    :return: returns a list of split rules
    """
    X, y = divide_data(data)
    dt_classifier = DecisionTreeClassifier(criterion, max_depth=max_features)
    dt_classifier.fit(X, y)
    rules = tree_walk_aux(dt_classifier.tree_, data.keys().drop(pred_label))
    return rules


def get_strongest_features_by_dt(data, max_features=2, label=pred_label):
    """
    :param data:
    :param max_features: max_features to extract
    :param label:
    :return: returns the firsts split rules of decision tree for every label e.g {Whites: ['Yearly_IncomeK <= -1.03',...]}
            the list assigned with the key can contain the rule "leaf" which means there were no more splits.
    """
    strongest_rules = {}
    for label_value in data[label].unique():
        binary_data = to_binary_label(data, label_value, label)
        strongest_rules[label_value] = get_dt_split_rules(binary_data, max_features)
    return strongest_rules


def choose_value(_min, _max, rule):
    code = "{} {}".format(_max, rule.split(" ", 1)[1])
    if not eval(code):
        return _max
    return _min


def fill_real_value(df, predicate, feature, shift=0.50):
    df.loc[predicate, feature] += abs(df.loc[predicate, feature]) * shift


def manipulate(data, rule):
    """
    :param data:
    :param rule: a string with the form 'x >= y' such that x is a feature and y is a value (threshold) generated by dt
    :return:
    """
    manipulated = data.copy()
    feature = rule.split(" ")[0]
    code = "manipulated['{}'] {}".format(feature, rule.split(' ', 1)[1])
    predicate = eval(code)
    if feature in real_num_features:
        fill_real_value(manipulated, predicate, feature)
    else:
        max_value = max(data[feature].values)
        min_value = min(data[feature].values)
        fill_value = choose_value(min_value, max_value, rule)
        manipulated.loc[predicate, feature] = fill_value
    return manipulated


def get_score(y_pred, value):
    return 100 * np.count_nonzero(y_pred == value) / np.size(y_pred)


def plot_before_after_and_save(heights, title, notes, dir_path):
    x = ["before", "after"]
    plt.title(title)
    plt.text(-0.1, -10, notes)
    plt.bar(x, heights)
    # plt.ylim([0, 100])
    fig_name = "{}-{}".format(title, notes)
    fig_path = os.path.join(dir_path, fig_name)
    if not os.path.exists(dir_path):
        logging.info("Could not save plots, path: {} does not exist".format(dir_path))
        raise ValueError
    plt.savefig(fig_path)
    plt.clf()


def get_file_name_datetime():
    return str(datetime.datetime.now()).replace(":", "-")


def prepare_plot_dirs():
    dirs_path = os.path.join(os.getcwd(), *["plots", get_file_name_datetime()])
    os.makedirs(dirs_path)
    return dirs_path


def get_plot_text(label, rule):
    return "{} vote precentage".format(label), "{} manipulation".format(rule.split(" ")[0])


def test_manipulation(train_data, val_data, model, label_rules_dict, verbose=False):
    """
    :param train_data:
    :param val_data:
    :param model:
    :param label_rules_dict: a dict with the form {'Yellows': [<dt_split1>, <dt_split2>,...]}, the list is the splits
           preformed by the dt classifier
    :param verbose: the logging info prints, if true the info will be written to the log and plotting will be created
    :return:
    """
    X_train, y_train = divide_data(train_data)
    X_val, _ = divide_data(val_data)

    dirs_path = None
    if verbose:
        dirs_path = prepare_plot_dirs()
    model.fit(X_train, y_train)
    y_pred_before = model.predict(X_val)
    best_diff = 0
    best_rule = None
    for label, rules in label_rules_dict.items():
        manipulated_val = manipulate(val_data, rules[0])
        X_val_man, _ = divide_data(manipulated_val)
        y_pred_after = model.predict(X_val_man)
        before_score = get_score(y_pred_before, label)
        after_score = get_score(y_pred_after, label)
        curr_diff = abs(before_score - after_score)
        if curr_diff > best_diff:
            best_diff = curr_diff
            best_rule = rules[0]
        if not verbose:
            continue
        logging.info(
            "--testing manipulation for: {} - rule: {} - before: {:.3f} - after: {:.3f}".format(label, rules[0],
                                                                                                before_score,
                                                                                                after_score))
        # title, notes = get_plot_text(label, rules[0])
        # plot_before_after_and_save([before_score, after_score], title, notes, dirs_path)

    manipulated_val = manipulate(val_data, best_rule)
    logging.info("With manipulation on '{}' the wining party is: {}".format(best_rule.split(" ")[0],
                                                                            get_predicted_major_party(manipulated_val,
                                                                                                      model)))


def save_old_log(overwrite=False):
    """
    this function saves the prev run's log in 'log-history' folder, it is called before the main and
    the 'logging.config' execute
    :param overwrite: whether you want to overwrite the prev log
    :return:
    """
    if overwrite:
        return
    log_old_path = os.path.join(os.getcwd(), "hw4.log")
    if not os.path.exists(log_old_path):
        return
    log_history_dir = os.path.join(os.getcwd(), "log-history")
    os.makedirs(log_history_dir, exist_ok=True)
    log_new_path = os.path.join(log_history_dir, "hw4.log")
    os.rename(log_old_path, log_new_path)
    new_name = "{}.log".format(get_file_name_datetime())
    os.rename(log_new_path, os.path.join(log_history_dir, new_name))


def get_nb_distance_dictionary(parties, df):
    """
    :param parties: dictionary of the shape: parties[name_of_party] = (array of center gaussian, naive_bayes_model)
    :return: returns dictionary of the shape dst_dict[(party_1, party_2)] with the euclidean distance between two parties.
    """
    dst_dict = dict()
    for _party_1, _party_2 in itertools.combinations(df['Vote'].unique(), 2):
        _mean_vector_party_1 = parties[_party_1][0]
        _mean_vector_party_2 = parties[_party_2][0]
        dst_dict[(_party_1, _party_2)] = distance.euclidean(_mean_vector_party_1, _mean_vector_party_2)
        dst_dict[(_party_2, _party_1)] = dst_dict[(_party_1, _party_2)]
    return dst_dict


def load_raw_and_prepare():
    """
    i've added this function so we dont need to change some bool vars in order to prepare/not prepare data.
    it is done automatically (checks whether 'prepared' folder exists and so on)
    :return:
    """
    logging.info('Loading raw data frames... ')
    raw_df = pd.read_csv('./csv/raw/ElectionsData.csv')
    logging.info('Done.\n')
    dirs_path_list = ['csv', 'prepared']
    prepared_path = os.path.join(os.getcwd(), *dirs_path_list)
    if os.path.exists(prepared_path):
        if len(os.listdir(prepared_path)) > 0:
            logging.info("Data already prepared, to re-prepare delete the prepared dir\n")
            return
    else:
        os.makedirs(prepared_path)
    logging.info("Preparing Data...")
    raw_df_train, raw_df_valid, raw_df_test = split_data_frame(raw_df)
    create_prepared_csv_dfs(raw_df_train, raw_df_valid, raw_df_test)


def load_prepared_data():
    logging.info('loading prepared data...')
    df_train = pd.read_csv('./csv/prepared/prepared_train.csv')
    df_valid = pd.read_csv('./csv/prepared/prepared_valid.csv')
    df_test = pd.read_csv('./csv/prepared/prepared_test.csv')
    return df_train, df_valid, df_test


def strongest_feature_info(df_train, verbose=False):
    """
    :param df_train:
    :param verbose: prints controller - if true the log will contain info about dt rules for each label
    :return: a dict which the key is a label and the value is a list of dt rules e.g {'Yellows': ['Will_...' >= 0.5, ...]}
    """
    label_rule_dict = get_strongest_features_by_dt(df_train)
    for l, rules in label_rule_dict.items():
        if not verbose:
            continue
        logging.info("--{}: {}".format(l, rules))
    return label_rule_dict


def tune_and_save_models(df_train):
    tuned_list_path = os.path.join(os.getcwd(), "tuned_list.pkl")
    if os.path.exists(tuned_list_path):
        logging.info("Tuned models exists, to re-construct models delete tuned_list.pkl file")
        return
    logging.info('Generating models...')
    models_to_tune = {"KNN's": generate_knn_models(), "DT's": generate_dt_models(),
                      "R-Forrest's": generate_rand_forest_models()}
    logging.info('Tuning hyper perameters...')
    tuned_list = choose_hyper_parameter_for_all(df_train, models_to_tune)
    with open('tuned_list.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(tuned_list, f)


def load_tuned_models():
    tuned_list_path = os.path.join(os.getcwd(), "tuned_list.pkl")
    if not os.path.exists(tuned_list_path):
        logging.info("You're trying to load models but the models file does not exist")
        return
    with open('tuned_list.pkl', 'rb') as f:
        return pickle.load(f)


def get_coalition_votes(coalition, model, df_train, df_val):
    """
    :param coalition: the coalition chosen
    :param model:
    :param df_train:
    :param df_val:
    :return: return the votes precentage
    """
    X_train, y_train = divide_data(df_train)
    X_val, _ = divide_data(df_val)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    coalition_chunk = np.count_nonzero(np.in1d(y_pred, np.array(coalition)))
    return 100 * coalition_chunk / np.size(y_pred)


def get_coalition(df_train, df_val):
    clusters_to_check = {'kMeans': generate_kmeans_models(), 'GMM': generate_gmm_models()}
    clusters = get_clusters(df_train, clusters_to_check)
    possible_coalitions = get_possible_coalitions(df_train, clusters)

    return get_most_homogeneous_coalition(df_val, possible_coalitions)


def manipulate_coalitions(curr_coal, label_rule_dict, model, df_train, df_val):
    """
    the idea is to take the opposition parties dt rules and to manipulate the data according to their first split
    (because i saw that manipulating the opposition dt rules increses the votes)
    for now it doesnt work :( i dont know whether the idea is wrong or i just dont know how to use your functions
    :param curr_coal: original coalition (without manipulation)
    :param label_rule_dict: a dict with the form {'Yellows': [<dt_split1, dt_split2, ...>]}
    :param model:
    :param df_train:
    :param df_val:
    :return:
    """
    op_list = label_rule_dict.keys()
    # op_list = [p for p in parties_list if p not in curr_coal]
    X_train, y_train = divide_data(df_train)
    model.fit(X_train, y_train)
    manipulated_train = df_train
    manipulated_val = df_val
    man_features = []
    best_coalition_votes = get_coalition_votes(curr_coal, model, df_train, df_val)
    for party in op_list:
        for i in range(2):
            curr_manipulated_train = manipulate(manipulated_train, label_rule_dict[party][i])
            X, y = divide_data(curr_manipulated_train)
            curr_manipulated_train[pred_label] = model.predict(X)
            curr_manipulated_val = manipulate(manipulated_val, label_rule_dict[party][i])
            X, y = divide_data(manipulated_val)
            curr_manipulated_val[pred_label] = model.predict(X)
            curr_coalition_votes = get_coalition_votes(curr_coal, model, df_train, curr_manipulated_val)
            if curr_coalition_votes >= best_coalition_votes:
                man_features.append(label_rule_dict[party][i])
                best_coalition_votes = curr_coalition_votes
                manipulated_train = curr_manipulated_train
                manipulated_val = curr_manipulated_val
    new_coal = get_coalition(manipulated_train, manipulated_val)
    score = get_coalition_votes(new_coal[1], model, manipulated_train, manipulated_val)
    logging.info("The new coalition is {}, with {} votes".format(new_coal, score))


def get_coalition_list_generative(party_name, d_dict, model, df_train, df_valid):
    """
    the idea is to get list of closest parties to party_name that can gather a coalition.
    we measure distance by the euclidean distance from one party gaussian center to another.
    these distances are given in d_dict.
    :param party_name: e.g. "Whites"
    :param d_dict: naive_bayes_distance_dictionary of the shape dst_dict[(party_1, party_2)] = real number.
    :param model: a classifier
    :param df_train
    :param df_val
    :return:
    """
    coalition_list = []
    parties_list = df_train['Vote'].unique()
    aux_list = [(_px, d_dict[party_name, _px]) for _px in parties_list if _px != party_name]
    aux_list.sort(key=lambda tup: -tup[1])
    coalition_list.append(party_name)
    coalition_size = get_coalition_votes(coalition_list, model, df_train, df_valid)
    while coalition_size < 51:
        coalition_list.append(aux_list.pop()[0])
        coalition_size = get_coalition_votes(coalition_list, model, df_train, df_valid)
    coalition_list.sort()
    return coalition_list, coalition_size


def get_parties_gaussian_mean_dictionary(df):
    """
    the idea is to use GaussianNB to grab the mean of the gaussian that represents each party.
    :param df:
    :return: parties: a dictionary of the shape parties["party_name"] = mean vector to the mean point of the party
    gaussian
    """
    parties = dict()
    for _party in df['Vote'].unique():
        _one_hot_df = to_binary_label(df, _party, 'Vote')
        X, y = divide_data(_one_hot_df)
        _model = GaussianNB()
        _model.fit(X, y)
        _index = list(_model.classes_).index(True)
        _mean_vector = _model.theta_[_index]
        parties[_party] = _mean_vector
    return parties


def get_possible_coalitions_generative(distance_dictionary, model, df_train, df_valid):
    possible_coalitions_generative = dict()
    for _party_name in df_train['Vote'].unique():
        possible_coalitions_generative["{}_based_coalition".format(_party_name)] = \
            get_coalition_list_generative(_party_name, distance_dictionary, model, df_train, df_valid)
    return possible_coalitions_generative


def filter_possible_coalitions_generative(possible_coalitions_generative):
    """
    the idea is to filter out coalitions that apear more then one time.
    :param possible_coalitions_generative:
    :return:
    """
    filtered_possible_coalitions_generative = dict()
    for _coalition_name, (_coalition_list, _) in possible_coalitions_generative.items():
        _coalition_list.sort()
        if _coalition_list not in filtered_possible_coalitions_generative.values():
            filtered_possible_coalitions_generative[_coalition_name] = _coalition_list
    return filtered_possible_coalitions_generative


def get_coalition_generative_method(df_train, df_valid, model):
    """ The idea is as following:
    1.  For each party train naive bayes with "one hot" train data.
    2.  Get gaussian center point from trained classifiers for each party.
    3.  For each point compute distance from any other point.
    4.  Build possible coalitions (minimal). close parties can establish a coalition.
    5.  Choose coalition which is most homogeneous - compute sum of all feature variance.
        leads to lowest result -> homogeneous coalition.
    :param df_train:
    :param df_valid:
    :param model: a classifier.
    :return: name and list of coalition.
    """
    parties_gaussian_mean_dictionary = get_parties_gaussian_mean_dictionary(df_train)
    naive_bayes_distance_dictionary = get_nb_distance_dictionary(parties_gaussian_mean_dictionary, df_train)
    possible_coalitions_generative = get_possible_coalitions_generative(naive_bayes_distance_dictionary, model,
                                                                        df_train, df_valid)
    filtered_possible_coalitions_generative = filter_possible_coalitions_generative(possible_coalitions_generative)
    return get_most_homogeneous_coalition(df_valid, filtered_possible_coalitions_generative)


def main():
    load_raw_and_prepare()
    df_train, df_valid, df_test = load_prepared_data()

    label_rule_dict = strongest_feature_info(df_train, verbose=False)

    tune_and_save_models(df_train)
    tuned_list = load_tuned_models()

    name, model, score = choose_model(tuned_list, df_train, df_valid)

    test_manipulation(df_train, df_valid, model, label_rule_dict, verbose=True)

    coal_name, coal_list = get_coalition(df_train, df_valid)
    logging.info(
        "Coalition acc. to clustering methods: {} - votes piece {}".format(
            coal_list, get_coalition_votes(coal_list, model, df_train, df_valid)))

    coal_name_generative, coal_list_generative = get_coalition_generative_method(df_train, df_valid, model)
    logging.info(
        "Coalition acc. to generative method: {} - votes piece {}".format(
            coal_list_generative, get_coalition_votes(coal_list_generative, model, df_train, df_valid)))

    manipulate_coalitions(coal_list, label_rule_dict, model, df_train, df_valid)
    pass


if __name__ == '__main__':
    save_old_log(overwrite=True)
    logging.basicConfig(filename='hw4.log', level=logging.INFO, format='%(levelname)s:%(message)s', filemode='w')
    main()