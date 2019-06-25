from data_infrastructure import *
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

import warnings

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

    kf = RepeatedStratifiedKFold(n_splits=k_fold)
    for train_i, test_i in kf.split(x_train, y_train):
        assert len(x_train.values[train_i]) == len(y_train[train_i])
        _clusters = model_class.fit_predict(x_train.values[train_i])
        assert len(_clusters) == len(x_train.values[train_i])

        # Internal metrics
        sil_score = metrics.silhouette_score(x_train.values[train_i], _clusters, metric='euclidean')
        silhouette_scores.append(sil_score)

        calisnky_harabaz = metrics.calinski_harabaz_score(x_train.values[train_i], _clusters)
        calinski_harabaz_scores.append(calisnky_harabaz)

        # External metrics
        homogeneity_score = metrics.homogeneity_score(y_train[train_i], _clusters)
        homogeneity_scores.append(homogeneity_score)
        completeness_score = metrics.completeness_score(y_train[train_i], _clusters)
        completeness_scores.append(completeness_score)
        v_score = metrics.v_measure_score(y_train[train_i], _clusters)
        v_measure_scores.append(v_score)

        # Relative metrics
        ari = metrics.adjusted_rand_score(y_train[train_i], _clusters)
        adjusted_rand_scores.append(ari)
        ami = metrics.adjusted_mutual_info_score(y_train[train_i], _clusters, average_method='arithmetic')
        adjusted_mutual_info_scores.append(ami)

    cluster_measures = {'homogeneity': np.mean(homogeneity_scores),
                        'completeness': np.mean(completeness_scores),
                        'v_measure': np.mean(v_measure_scores),
                        'adjusted_rand': np.mean(adjusted_rand_scores),
                        'mutual_info': np.mean(adjusted_mutual_info_scores),
                        'silhouette': np.mean(silhouette_scores),
                        'CLH': np.mean(calinski_harabaz_scores),
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
    plt.plot(n_component, k_comparisons['homogeneity'])
    plt.xlabel('number of components')
    plt.ylabel('homogeneity score')
    plt.grid()
    plt.subplot(132)
    plt.plot(n_component, k_comparisons['completeness'])
    plt.xlabel('number of components')
    plt.ylabel('completeness score')
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
    plt.subplot(121)
    plt.plot(n_component, k_comparisons['CLH'])
    plt.xlabel('number of components')
    plt.ylabel('Calinsky-Harabaz score')
    plt.grid()
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


def main():
    train_df, val_df, test_df = import_from_csv(TRAIN_PATH), import_from_csv(VALIDATION_PATH), import_from_csv(TEST_PATH)
    test_unlabeled_df = import_from_csv(TEST_UNLABELED_PATH)

    x_train, y_train = divide_data(train_df)
    test_gmm(x_train, y_train)


if __name__ == '__main__':
    main()
