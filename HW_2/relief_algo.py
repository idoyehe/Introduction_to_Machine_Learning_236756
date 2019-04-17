"""
This algorithm assumption is that there are no Nan values in the given data
"""

from data_infrastructure import *
from sklearn.preprocessing import MinMaxScaler


def relief(x_train: DataFrame, y_train: DataFrame, local_nominal_feature: list,
           local_numerical_features: list, num_of_iter, threshold) -> list:
    """
    This is the main algorithem, call it to get the list of features whose avg
    score is bigger then the threshold
    :param raw_data: The data to work on
    :param label_name: The label name that should be predicted
    :param num_of_iter: number of iterations to run the algorithem
    :param threshold: The threshold that distinguish between good and bad
    features
    :return: List of features that thier avg score is bigger then the threshold
    a.k.a the "good" features
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train[local_numerical_features] = scaler.fit_transform(
        x_train[local_numerical_features])
    weight_features_dict = {}
    features = x_train.columns.values
    for f in features:
        weight_features_dict[f] = 0

    for i in range(num_of_iter):
        x = x_train.sample(n=1)
        x_index = x.index[0]
        x_label = y_train[x_index]

        same_class_data = x_train[y_train == x_label]
        same_class_data = same_class_data.drop(index=x_index)

        different_class_data = x_train[y_train != x_label]

        nearest_hit = closest_fit(same_class_data, x, local_nominal_feature,
                                  local_numerical_features)
        nearest_miss = closest_fit(different_class_data, x,
                                   local_nominal_feature,
                                   local_numerical_features)

        for nom_f in local_nominal_feature:
            weight_features_dict[nom_f] += not x[nom_f].values[0] == \
                                               x_train[nom_f].values[
                                                   nearest_miss]
            weight_features_dict[nom_f] -= not x[nom_f].values[0] == \
                                               x_train[nom_f].values[
                                                   nearest_hit]

        for num_f in local_numerical_features:
            weight_features_dict[num_f] += (x[num_f].values[0] -
                                            x_train[num_f].values[
                                                nearest_miss]) ** 2
            weight_features_dict[num_f] -= (x[num_f].values[0] -
                                            x_train[num_f].values[
                                                nearest_hit]) ** 2

    for f in features:
        weight_features_dict[f] = weight_features_dict[f] / num_of_iter
    chosen_set = [f for f in features if weight_features_dict[f] > threshold]
    return chosen_set
