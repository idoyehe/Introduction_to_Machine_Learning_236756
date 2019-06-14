from data_infrastructure import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from scipy.spatial.distance import euclidean
from itertools import combinations
from sklearn.metrics import accuracy_score


def get_dt_split_rules(data):
    x_train, y_train = divide_data(data)
    rf_classifier = RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=3, min_samples_leaf=1, n_estimators=500)
    rf_classifier.fit(x_train, y_train)
    temp_feature_importances = []
    for feature_index, feature_name in enumerate(selected_features_without_label):
        temp_feature_importances.append((feature_name, rf_classifier.feature_importances_[feature_index]))
    temp_feature_importances.sort(key=lambda tup: -tup[1])

    feature_importances = {}
    for tup in temp_feature_importances:
        feature_importances[tup[0]] = tup[1]

    return feature_importances


def get_strongest_features_by_dt(data):
    """
    :param data: dataframe
    :return: returns the firsts split rules of decision tree for every label e.g {Whites: ['Yearly_IncomeK <= -1.03',...]}
            the list assigned with the key can contain the rule "leaf" which means there were no more splits.
    """
    strongest_features = {}
    for label_value in num2label.keys():
        binary_data = to_binary_class(data, label_value)
        strongest_features[label_value] = get_dt_split_rules(binary_data)

    for _label, _features in strongest_features.items():
        print(f"strongest features of {_label} is: {_features}")
    return strongest_features


def choose_hyper_parameter(models, x, y, kfolds: int = 5):
    best_score = float('-inf')
    best_model = None
    for _name, _model in models:
        _score = score(x_train=x, y_train=y, clf=_model, k=kfolds)
        if _score > best_score:
            best_score = _score
            best_model = _name, _model, best_score
    return best_model


def generate_kmeans_models():
    for num_of_clusters in [2, 3, 4]:
        model_name = f"KMean_with_{num_of_clusters}_clusters"
        cluster = KMeans(num_of_clusters, max_iter=2500, random_state=0)
        yield model_name, cluster


def generate_gmm_models():
    for num_of_clusters in [2, 3, 4]:
        cluster_name = f"GMM_with_{num_of_clusters}_clusters"
        cluster = GaussianMixture(n_components=num_of_clusters, max_iter=2500, init_params='random', random_state=0)
        yield cluster_name, cluster


def evaluate_clusters_models(df_train, models_generator, label_in_cluster_threshold: float = 0.60, k_fold: int = 5):
    best_model = None
    largest_cluster_size = 0
    kf = KFold(k_fold, random_state=0)
    for model_name, model_class in models_generator:
        clusters_size = 0
        for train_i, test_i in kf.split(df_train):
            x_train, y_train = divide_data(df_train.iloc[train_i, :])
            x_test, y_test = divide_data(df_train.iloc[test_i, :])
            _model_fitted = model_class.fit(x_train)
            _cluster_label = _model_fitted.predict(x_test)
            _labels_in_cluster = None
            for cluster in set(_cluster_label):
                _samples_in_cluster = y_test[_cluster_label == cluster]  # all the samples in the cluster
                _labels_in_cluster = list(set(_samples_in_cluster))  # all the labels that appears in cluster

                for _label in tuple(_labels_in_cluster):
                    _label_in_total = sum(y_test == _label)
                    _label_in_cluster = sum(_samples_in_cluster == _label)
                    _label_percent_in_cluster = _label_in_cluster / _label_in_total  # percent from this label in cluster higher is better
                    if _label_percent_in_cluster < label_in_cluster_threshold:
                        _labels_in_cluster.remove(_label)

                clusters_size += len(_labels_in_cluster)  # how many labels is actual in the cluster

        if clusters_size > largest_cluster_size:
            largest_cluster_size = clusters_size
            best_model = model_name, model_class

    return best_model


def get_possible_clustered_coalitions(df_train: DataFrame, x_val, y_val, clusters_to_check):
    x_train, y_train = divide_data(df_train)
    _vote_results = get_sorted_vote_division(y_val)
    possible_coalitions = {}
    for model_name, model_class in clusters_to_check.items():
        model_class = model_class.fit(x_train)
        clusters_per = model_class.predict(x_val)

        for group in set(clusters_per):
            _group_votes = y_val[clusters_per == group]
            _parties_in_group = list(set(_group_votes))
            # remove parties that are not really in group (less then 80%)
            # party is in group if more then 90% of its voters are in group.
            for _party in tuple(_parties_in_group):
                _party_votes = sum(y_val == _party)
                _party_voters_in_group = sum(_group_votes == _party)
                _group_voters_proportion = _party_voters_in_group / _party_votes
                if _group_voters_proportion < global_transportation_threshold:
                    _parties_in_group.remove(_party)
            _group_size = sum([_vote_results[p] for p in _parties_in_group])
            if _group_size > 0.51:
                possible_coalitions[f'{model_name}_group-{group}'] = _parties_in_group

    return filter_possible_coalitions(possible_coalitions)


def gaussian_nb_hyperparametrs_tuning(df_train: DataFrame, k_fold: int = 5):
    guassien_naive_base = (
        GaussianNB(var_smoothing=1e-7),
        GaussianNB(var_smoothing=1e-8),
        GaussianNB(var_smoothing=1e-9),
        GaussianNB(var_smoothing=1e-10)
    )
    x_train, y_train = divide_data(df_train)
    best_score = float('-inf')
    best_clf = None
    for clf in guassien_naive_base:
        _score = score(x_train=x_train, y_train=y_train, clf=clf, k=k_fold)
        if _score > best_score:
            best_score = _score
            best_clf = clf

    return best_clf, best_score


def qda_hyperparametrs_tuning(df_train: DataFrame, k_fold: int = 5):
    qda_clfs = (
        QDA(),
        QDA(priors=None, reg_param=0., store_covariance=True, tol=1.0e-3),
        QDA(priors=None, reg_param=0., store_covariance=True, tol=1.0e-4),
        QDA(priors=None, reg_param=0., store_covariance=True, tol=1.0e-5)
    )
    x_train, y_train = divide_data(df_train)
    best_score = float('-inf')
    best_clf = None
    for clf in qda_clfs:
        _score = score(x_train=x_train, y_train=y_train, clf=clf, k=k_fold)
        if _score > best_score:
            best_score = _score
            best_clf = clf

    return best_clf, best_score


def labels_generative_mean(df, model):
    """
    use GaussianNB to grab the mean of the gaussian of each label.
    :param df:data frame
    :param model: the model if GNB
    :return: label_mean_dict: a dictionary of the shape parties["label"] = mean vector to the mean point of the party
    gaussian
    """
    label_mean_dict = dict()
    for _party in num2label.keys():
        _one_hot_df = to_binary_class(df, _party)
        x_train, y_train = divide_data(_one_hot_df)
        model.fit(x_train, y_train)
        _index = list(model.classes_).index(True)

        if hasattr(model, "theta_"):
            _mean_vector = model.theta_[_index]
        else:
            _mean_vector = model.means_[_index]

        label_mean_dict[_party] = _mean_vector
    return label_mean_dict


def labels_distance_dictionary(labels_mean_dictionary):
    """
    :param labels_mean_dictionary: dictionary of the shape: dict[label] = array of center of probabilities
    :return: returns dictionary of the shape dst_dict[(label_1, label_2)] with the euclidean distance between two parties.
    """
    dst_dict = dict()
    for _label_1, _label_2 in combinations(num2label.keys(), 2):
        _mean_vector_label_1 = labels_mean_dictionary[_label_1]
        _mean_vector_label_2 = labels_mean_dictionary[_label_2]
        dst_dict[(_label_1, _label_2)] = euclidean(_mean_vector_label_1, _mean_vector_label_2)
        dst_dict[(_label_2, _label_1)] = dst_dict[(_label_1, _label_2)]
    return dst_dict


def build_coalition_using_generative_data(y_test, parties_mean_dictionary):
    """ The idea is as following:
    1.  Get center point from trained classifiers for each party.
    2.  For each point compute distance from any other point.
    3.  Build possible coalitions (minimal). close parties can establish a coalition.
    :param y_test: labels of data
    :param parties_mean_dictionary: the mean vector of the label
    :return: name and list of coalition.
    """
    distance_dictionary = labels_distance_dictionary(parties_mean_dictionary)
    possible_coalitions_generative = get_possible_coalitions_generative(distance_dictionary, y_test)
    return filter_possible_coalitions(possible_coalitions_generative)


def get_possible_coalitions_generative(distance_dictionary, y_test):
    possible_coalitions_generative = {}
    for _label, label_index in label2num.items():
        possible_coalitions_generative[f"{_label}_based_coalition"] = \
            get_coalition_list_generative(y_test, label_index, distance_dictionary)
    return possible_coalitions_generative


def get_coalition_size(y_test, coalition):
    """
    :param y_test: labels of data
    :param coalition: requested coalition
    :return:
    """
    coalition_chunk = np.count_nonzero(np.in1d(y_test, np.array(coalition)))
    return coalition_chunk / np.size(y_test)


def get_coalition_list_generative(y_test, ref_label_index, d_dict):
    """
    :param y_test: labels of data
    :param ref_label_index: the label which is the center of the coalition
    :param d_dict: distance dict between labels
    :return: the coalition and coalition size
    """
    coalition_list = []
    parties_list = num2label.keys()
    aux_list = [(_l, d_dict[ref_label_index, _l]) for _l in parties_list if _l != ref_label_index]
    aux_list.sort(key=lambda tup: tup[1])
    coalition_list.append(ref_label_index)
    coalition_size = get_coalition_size(y_test, coalition_list)
    while coalition_size < 0.51:
        coalition_list.append(aux_list.pop()[0])
        coalition_size = get_coalition_size(y_test, coalition_list)
    coalition_list.sort()
    return coalition_list


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
    :param features: features to show
    :param coalition_feature_variance: the variance of each feature
    :return:
    """
    plt.barh(features, coalition_feature_variance)
    plt.title(title)
    plt.show()


def increase_coalition(df):
    manipulated = df.copy()
    manipulated["Weighted_education_rank"] = -0.292
    manipulated["Avg_Residancy_Altitude"] = -0.39985
    return manipulated


def decrease_coalition(df):
    manipulated = df.copy()
    manipulated.loc[manipulated["Weighted_education_rank"] <= 0.095, "Weighted_education_rank"] = 0.2
    manipulated.loc[manipulated["Avg_monthly_expense_on_pets_or_plants"] > 1.531, "Avg_monthly_expense_on_pets_or_plants"] = 1.3
    return manipulated


def build_alternative_coalition(df_train: DataFrame, df_val: DataFrame, classifier):
    manipulated_df_train = decrease_coalition(df_train)
    manipulated_df_val = decrease_coalition(df_val)

    x_m_val, y_m_val = divide_data(manipulated_df_val)
    winner_color(classifier, x_m_val)

    clusters_to_check = get_clusters_models(manipulated_df_train)
    possible_coalitions = get_possible_clustered_coalitions(manipulated_df_train, x_m_val, y_m_val, clusters_to_check)
    coalition, coalition_feature_variance = get_most_homogeneous_coalition(manipulated_df_val, possible_coalitions)
    coalition_size = get_coalition_size(y_m_val, coalition[1])
    print(f"Alternative coalition using Cluster model is {coalition} with size of {coalition_size}")
    plot_feature_variance(selected_numerical_features, coalition_feature_variance)


def build_stronger_coalition(df_train: DataFrame, df_val: DataFrame):
    manipulated_df_train = increase_coalition(df_train)
    manipulated_df_val = increase_coalition(df_val)
    x_m_val, y_m_val = divide_data(manipulated_df_val)
    clusters_to_check = get_clusters_models(manipulated_df_train)

    possible_coalitions = get_possible_clustered_coalitions(manipulated_df_train, x_m_val, y_m_val, clusters_to_check)
    coalition, coalition_feature_variance = get_most_homogeneous_coalition(manipulated_df_val, possible_coalitions)
    coalition_size = get_coalition_size(y_m_val, coalition[1])
    print(f"Stronger coalition using Cluster model is {coalition} with size of {coalition_size}")
    plot_feature_variance(selected_numerical_features, coalition_feature_variance)


def get_clusters_models(df_train: DataFrame):
    clusters_to_check = {}
    cluster_name, cluster_model = evaluate_clusters_models(df_train, generate_kmeans_models())
    clusters_to_check[cluster_name] = cluster_model

    cluster_name, cluster_model = evaluate_clusters_models(df_train, generate_gmm_models())
    clusters_to_check[cluster_name] = cluster_model
    return clusters_to_check


def get_coalition_by_clustering(df_train: DataFrame, df_val: DataFrame, df_test: DataFrame, x_test, y_test):
    clusters_to_check = get_clusters_models(df_train)
    x_val, y_val = divide_data(df_val)
    possible_coalitions = get_possible_clustered_coalitions(df_train, x_val, y_val, clusters_to_check)
    coalition, coalition_feature_variance = get_most_homogeneous_coalition(df_val, possible_coalitions)
    coalition_size = get_coalition_size(y_val, coalition[1])
    print(f"coalition using {coalition[0]} is {coalition[1]} with size of {coalition_size}")
    plot_feature_variance(selected_numerical_features, coalition_feature_variance)

    possible_coalitions = get_possible_clustered_coalitions(df_train, x_test, y_test, clusters_to_check)
    coalition, coalition_feature_variance = get_most_homogeneous_coalition(df_test, possible_coalitions)
    coalition_size = get_coalition_size(y_test, coalition[1])
    print(f"TEST coalition using {coalition[0]} is {coalition[1]} with size of {coalition_size}")
    plot_feature_variance(selected_numerical_features, coalition_feature_variance)
    return clusters_to_check, coalition[1]


def get_coalition_by_generative(df_train: DataFrame, df_val: DataFrame, df_test: DataFrame, y_test):
    gaussian_nb_clf, gaussian_nb_score = gaussian_nb_hyperparametrs_tuning(df_train)
    qda_clf, qda_score = qda_hyperparametrs_tuning(df_train)

    labels_guassian_mean = labels_generative_mean(df_train, gaussian_nb_clf)
    labels_qda_mean = labels_generative_mean(df_train, qda_clf)

    x_val, y_val = divide_data(df_val)
    naive_base_coalitions = build_coalition_using_generative_data(y_val, labels_guassian_mean)
    qda_coalitions = build_coalition_using_generative_data(y_val, labels_qda_mean)

    coalition_nb, coalition_nb_feature_variance = get_most_homogeneous_coalition(df_val, naive_base_coalitions)
    coalition_nb_size = get_coalition_size(y_val, coalition_nb[1])
    print(f"coalition using Gaussian Naive Base model is {coalition_nb} with size of {coalition_nb_size}")
    plot_feature_variance(selected_numerical_features, coalition_nb_feature_variance)

    coalition_qda, coalition_qda_feature_variance = get_most_homogeneous_coalition(df_val, qda_coalitions)
    coalition_qda_size = get_coalition_size(y_val, coalition_qda[1])
    print(f"coalition using QDA model is {coalition_qda} with size of {coalition_qda_size}")
    plot_feature_variance(selected_numerical_features, coalition_qda_feature_variance)

    coalitions_generative = build_coalition_using_generative_data(y_test, labels_guassian_mean)
    coalitions_generative, coalition_feature_variance = get_most_homogeneous_coalition(df_test, coalitions_generative)
    coalitions_generative_size = get_coalition_size(y_test, coalitions_generative[1])
    print(f"TEST coalition using Gaussian Naive Base model is {coalitions_generative} with size of {coalitions_generative_size}")
    plot_feature_variance(selected_numerical_features, coalition_qda_feature_variance)

    return coalitions_generative


def main():
    df_train, df_val, df_test = load_prepared_dataFrames()
    plot_feature_variance(selected_numerical_features, df_train.var(axis=0)[selected_numerical_features], "Feature Variance")

    classifier = RandomForestClassifier(random_state=0, criterion='gini', min_samples_split=3, min_samples_leaf=1, n_estimators=500)
    x_train, y_train = divide_data(df_train)
    x_test, y_test = divide_data(df_test)
    classifier.fit(x_train, y_train)
    y_pred_test = classifier.predict(x_test)
    print(f"classifier accuracy score: {accuracy_score(y_true=y_test, y_pred=y_pred_test)}")

    get_coalition_by_clustering(df_train, df_val, df_test, x_test, y_pred_test)

    get_coalition_by_generative(df_train, df_val, df_test, y_pred_test)

    get_strongest_features_by_dt(df_train)

    build_stronger_coalition(df_train, df_val)

    build_alternative_coalition(df_train, df_val, classifier)


if __name__ == '__main__':
    main()
