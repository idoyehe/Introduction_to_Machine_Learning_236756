from data_infrastructure import *
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter, OrderedDict


import warnings

NUM_OF_CLUSTERS = 5

warnings.filterwarnings('ignore')


def generate_gmm_models(ranger):
    for num_of_clusters in ranger:
        cluster_name = f"GMM_with_{num_of_clusters}_clusters"
        cluster = GaussianMixture(n_components=num_of_clusters, covariance_type='full', init_params='random', random_state=0)
        yield cluster_name, cluster


def evaluate_gmm_model(x_train, y_train, model_class, k_fold: int = 3):
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []
    adjusted_rand_scores = []
    adjusted_mutual_info_scores = []
    silhouette_scores = []
    calinski_harabaz_scores = []
    davies_bouldin_scores = []
    jaccard_similarity_scores = []

    kf = RepeatedStratifiedKFold(n_splits=k_fold)
    for train_i, test_i in kf.split(x_train, y_train):
        assert len(x_train.values[train_i]) == len(y_train[train_i])
        _clusters = model_class.fit_predict(x_train.values[train_i])
        assert len(_clusters) == len(x_train.values[train_i])

        # Internal metrics
        sil_score = metrics.silhouette_score(x_train.values[train_i], _clusters, metric='euclidean')
        silhouette_scores.append(sil_score)

        davies_bouldin = metrics.davies_bouldin_score(x_train.values[train_i], _clusters)
        davies_bouldin_scores.append(davies_bouldin)
        # calisnky_harabaz = metrics.calinski_harabaz_score(x_train.values[train_i], _clusters)
        # calinski_harabaz_scores.append(calisnky_harabaz)

        # External metrics
        # homogeneity_score = metrics.homogeneity_score(y_train[train_i], _clusters)
        # homogeneity_scores.append(homogeneity_score)
        # completeness_score = metrics.completeness_score(y_train[train_i], _clusters)
        # completeness_scores.append(completeness_score)
        v_score = metrics.v_measure_score(y_train[train_i], _clusters)
        v_measure_scores.append(v_score)

        jaccard_similarity = metrics.jaccard_similarity_score(y_train[train_i], _clusters)
        jaccard_similarity_scores.append(jaccard_similarity)

        # Relative metrics
        ari = metrics.adjusted_rand_score(y_train[train_i], _clusters)
        adjusted_rand_scores.append(ari)
        ami = metrics.adjusted_mutual_info_score(y_train[train_i], _clusters, average_method='arithmetic')
        adjusted_mutual_info_scores.append(ami)

    # cluster_measures = {'homogeneity': np.mean(homogeneity_scores),
    #                     'completeness': np.mean(completeness_scores),
    #                     'v_measure': np.mean(v_measure_scores),
    #                     'adjusted_rand': np.mean(adjusted_rand_scores),
    #                     'mutual_info': np.mean(adjusted_mutual_info_scores),
    #                     'silhouette': np.mean(silhouette_scores),
    #                     'CLH': np.mean(calinski_harabaz_scores),
    #                     }

    cluster_measures = {'davies_bouldin': np.mean(davies_bouldin_scores),
                        'jaccard_similarity': np.mean(jaccard_similarity_scores),
                        'v_measure': np.mean(v_measure_scores),
                        'adjusted_rand': np.mean(adjusted_rand_scores),
                        'mutual_info': np.mean(adjusted_mutual_info_scores),
                        'silhouette': np.mean(silhouette_scores)
                        }

    return cluster_measures


def test_gmm(x_train: DataFrame, y_train: Series):
    k_comparisons = DataFrame()
    n_component = range(2, 20)

    for model_name, model_class in generate_gmm_models(n_component):
        results = evaluate_gmm_model(x_train, y_train, model_class)
        results = DataFrame([results])
        k_comparisons = concat([k_comparisons, results])

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(n_component, k_comparisons['davies_bouldin'])
    plt.xlabel('number of components')
    plt.ylabel('davies_bouldin score')
    plt.grid()
    plt.subplot(132)
    plt.plot(n_component, k_comparisons['jaccard_similarity'])
    plt.xlabel('number of components')
    plt.ylabel('jaccard_similarity score')
    plt.grid()
    plt.subplot(133)
    plt.plot(n_component, k_comparisons['v_measure'])
    plt.xlabel('number of components')
    plt.ylabel('v_measure score')
    plt.grid()
    plt.suptitle('External cluster validity measures of Gaussian Model Mixture '
                 'as a function of the number of components')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(122)
    plt.plot(n_component, k_comparisons['silhouette'])
    plt.xlabel('number of components')
    plt.ylabel('silhouette score')
    plt.grid()
    plt.suptitle('Internal cluster validity measures of Gaussian Model Mixture '
                 'as a function of the number of components')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(n_component, k_comparisons['adjusted_rand'])
    plt.xlabel('number of components')
    plt.ylabel('Adjusted Rand score')
    plt.grid()
    plt.subplot(122)
    plt.plot(n_component, k_comparisons['mutual_info'])
    plt.xlabel('number of components')
    plt.ylabel('Adjusted mutual information score')
    plt.grid()
    plt.suptitle('Relative cluster validity measures of Gaussian Model Mixture '
                 'as a function of the number of components')
    plt.show()

def generative_model_test(data: DataFrame, labels: Series):
    '''
    Tests a generative model, LDA, and plots validity measure graphs
    '''

    classifier = LinearDiscriminantAnalysis()

    num_parties = len(labels.unique())
    num_features = len(data.columns)

    fold_means = []
    k_fold = RepeatedStratifiedKFold(n_splits=5)
    for train_idx, test_idx in k_fold.split(data, labels):
        assert len(data.values[train_idx]) == len(labels[train_idx])
        classifier.fit(data.values[train_idx], labels[train_idx])
        fold_means.append(classifier.means_)

    # Mean
    party_means = []
    for i in range(num_parties):
        party_means.append([])
        curr_party_mean = party_means[i]
        for j in range(num_features):
            curr_sum = 0
            for mean in fold_means:
                curr_sum += mean[i][j]
            party_feature_average = curr_sum / len(fold_means)
            curr_party_mean.append(party_feature_average)

    # Asserts
    print(party_means)
    assert len(party_means) == num_parties
    for i in range(num_parties):
        assert len(party_means[i]) == num_features

    # Generate distance matrix between means
    mean_dist_mat = euclidean_distances(party_means, party_means)
    assert mean_dist_mat.shape == (num_parties, num_parties)
    mean_dist_mat = np.around(mean_dist_mat, decimals=3)

    # Plot distance matrix
    axis_positions = np.arange(num_parties)
    plt.figure(figsize=(12, 9))
    plt.imshow(mean_dist_mat, interpolation='nearest')
    predictions_to_labels = np.vectorize(lambda n: num2label[int(n)])
    y_test_prd = predictions_to_labels(labels)
    plt.xticks(axis_positions, np.unique(y_test_prd))
    plt.yticks(axis_positions, np.unique(y_test_prd))
    plt.colorbar()
    plt.grid(True)
    plt.title('Euclidean distances between means of parties, QDA')
    plt.tight_layout()
    plt.show()

def get_cluster_histogram(num_of_cluster, cluster_affilation, y_true):
    label = y_true
    hist = {}
    for index in range(len(cluster_affilation)):
        if cluster_affilation[index] == num_of_cluster:
            if not(num2label[label[index]] in hist):
                hist[num2label[label[index]]] = 1
            else:
                hist[num2label[label[index]]] += 1
    return hist

def print_cluster_distrebutions(num_of_clusters, y_pred, y_true):
    for i in range(num_of_clusters):
        d = get_cluster_histogram(i, y_pred, y_true)
        od = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        print(f"cluster {i} distribution: {od}")

def print_per_party_distrebution(num_of_clusters, y_pred, y_true: Series):
    predictions_to_labels = np.vectorize(lambda n: num2label[int(n)])
    y_test_prd = predictions_to_labels(y_true)
    votes_counter = Counter(y_true)
    clusters_distrebution_list = []
    for i in range(num_of_clusters):
        clusters_distrebution_list.append(get_cluster_histogram(i, y_pred, y_true))
    for party in votes_counter:
        party_label = num2label[party]
        party_size = votes_counter[party]
        party_dist = []
        for i in range(len(clusters_distrebution_list)):
            if party_label in clusters_distrebution_list[i]:
                size_of_party_in_cluster = clusters_distrebution_list[i][party_label]
                party_dist.append((i, size_of_party_in_cluster/party_size))
        print(f"{party_label} dist: {party_dist}")

def main():
    train_df, val_df, test_df = import_from_csv(TRAIN_PATH), import_from_csv(VALIDATION_PATH), import_from_csv(TEST_PATH)
    test_unlabeled_df = import_from_csv(TEST_UNLABELED_PATH)

    x_train, y_train = divide_data(train_df)
    # test_gmm(x_train, y_train)

    clf = GaussianMixture(n_components=NUM_OF_CLUSTERS, covariance_type='full', init_params='random', random_state=0)
    y_pred = clf.fit_predict(x_train)
    print_cluster_distrebutions(NUM_OF_CLUSTERS, y_pred, y_train)
    print_per_party_distrebution(NUM_OF_CLUSTERS, y_pred, y_train)

    print("Reds, Oranges, Greys")
    print("Green, Whites")
    print("Yellows, Blues")
    print("Khakis, Violets, Turqoises")
    print("Browns, Purples")
    print("Pink")
    generative_model_test(x_train, y_train)

    predictions_to_labels = np.vectorize(lambda n: num2label[int(n)])
    y_train_labels = predictions_to_labels(y_train)
    plt.hist(y_train_labels)
    plt.show()

if __name__ == '__main__':
    main()
