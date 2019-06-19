from data_infrastructure import *
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def get_possible_clustered_coalitions(df_train: DataFrame, x_test, y_test, cluste_model):
    x_train, y_train = divide_data(df_train)
    _vote_results = get_sorted_vote_division(y_test)
    possible_coalitions = {}
    model_class = cluste_model.fit(x_train)
    clusters_per = model_class.predict(x_test)

    for group in set(clusters_per):
        _group_votes = y_test[clusters_per == group]
        _parties_in_group = list(set(_group_votes))
        # remove parties that are not really in group (less then 80%)
        # party is in group if more then 90% of its voters are in group.
        for _party in tuple(_parties_in_group):
            _party_votes = sum(y_test == _party)
            _party_voters_in_group = sum(_group_votes == _party)
            _group_voters_proportion = _party_voters_in_group / _party_votes
            if _group_voters_proportion < global_party_in_coalition_threshold:
                _parties_in_group.remove(_party)
        _group_size = sum([_vote_results[p] for p in _parties_in_group])
        if _group_size > 0.51:
            possible_coalitions[f'Base_Cluster-{group}'] = _parties_in_group

    return filter_possible_coalitions(possible_coalitions)


def get_coalition_size(y_test, coalition):
    """
    :param y_test: labels of data
    :param coalition: requested coalition
    :return:
    """
    coalition_chunk = np.count_nonzero(np.in1d(y_test, np.array(coalition)))
    return coalition_chunk / np.size(y_test)


def get_coalition_variance(df, coalition_parties):
    df_coalition = df.loc[df[label].isin(coalition_parties), :]
    coalition_feature_variance = [np.var(df_coalition[f]) for f in selected_numerical_features]
    return coalition_feature_variance


def get_most_homogeneous_coalition(df: DataFrame, possible_coalitions):
    """
    :param df: dataframe
    :param possible_coalitions: possible coalition
    :return: the most homogeneous coalition and it's feature variance vector
    """
    max_total_variance = float('inf')
    best_coalition_feature_variance = None
    for _coalition_name, _coalition_parties in possible_coalitions.items():
        _coalition_feature_variance = get_coalition_variance(df, _coalition_parties)
        if sum(_coalition_feature_variance) < max_total_variance:
            max_total_variance = sum(_coalition_feature_variance)
            most_homogeneous_coalition_parties = (_coalition_name, _coalition_parties)
            best_coalition_feature_variance = _coalition_feature_variance
    return most_homogeneous_coalition_parties, best_coalition_feature_variance


def plot_feature_variance(features, coalition_feature_variance, title="Coalition Feature Variance"):
    """
    :param title: title of the graph
    :param features: features to show
    :param coalition_feature_variance: the variance of each feature
    :return:
    """
    plt.barh(features, coalition_feature_variance)
    plt.title(title)
    plt.show()


def get_coalition_by_clustering(df_train: DataFrame, df_val: DataFrame, df_test: DataFrame, x_test, y_test):
    cluster_model = GaussianMixture(n_components=2, max_iter=2500, init_params='random', random_state=0)

    x_val, y_val = divide_data(df_val)
    possible_coalitions = get_possible_clustered_coalitions(df_train, x_val, y_val, cluster_model)
    coalition, coalition_feature_variance = get_most_homogeneous_coalition(df_val, possible_coalitions)
    coalition_size = get_coalition_size(y_val, coalition[1])
    print(f"Simulate Coalition using {coalition[0]} is {coalition[1]} with size of {coalition_size}")
    plot_feature_variance(selected_numerical_features, coalition_feature_variance)

    # df_test[label] = y_test
    #
    # possible_coalitions = get_possible_clustered_coalitions(df_train, x_test, y_test, clusters_to_check)
    # coalition, coalition_feature_variance = get_most_homogeneous_coalition(df_test, possible_coalitions)
    # coalition_size = get_coalition_size(y_test, coalition[1])
    # print(f"TEST coalition using {coalition[0]} is {coalition[1]} with size of {coalition_size}")
    # plot_feature_variance(selected_numerical_features, coalition_feature_variance)
    # return clusters_to_check, coalition[1]


def main():
    train_df = import_from_csv(TRAIN_PATH)
    valid_df = import_from_csv(VALIDATION_PATH)
    test_df = import_from_csv(TEST_PATH)

    plot_feature_variance(selected_numerical_features, train_df.var(axis=0)[selected_numerical_features], "Feature Variance")

    x_test = test_df[selected_features_without_label]
    y_pred_test = import_from_csv(EXPORT_TEST_PREDICTIONS)
    predictions_to_number = np.vectorize(lambda _label: label2num[_label])
    y_pred_test = predictions_to_number(y_pred_test[label])

    test_df[label] = y_pred_test

    get_coalition_by_clustering(train_df, valid_df, test_df, x_test, y_pred_test)


if __name__ == '__main__':
    main()
