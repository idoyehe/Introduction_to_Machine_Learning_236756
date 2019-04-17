"""
This algorithm assumption is that there are no Nan values in the given data
"""

from data_infrastructure import *
from sklearn.preprocessing import MinMaxScaler


def relief(x_train: DataFrame, y_train: DataFrame, local_nominal_feature: list, local_numerical_features: list, num_of_iter,
           threshold) -> list:
    """
    :param x_train: train data frame
    :param y_train: train labels data frame
    :param local_nominal_feature: the nominal features to examine
    :param local_numerical_features: the numerical features to examine
    :param num_of_iter: number of iterations
    :param threshold: threshold
    :return: relif selected features
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
            weight_features_dict[nom_f] += x[nom_f].values[0] != x_train[nom_f].values[nearest_miss]
            weight_features_dict[nom_f] -= x[nom_f].values[0] != x_train[nom_f].values[nearest_hit]

        for num_f in local_numerical_features:
            weight_features_dict[num_f] += (x[num_f].values[0] - x_train[num_f].values[nearest_miss]) ** 2
            weight_features_dict[num_f] -= (x[num_f].values[0] - x_train[num_f].values[nearest_hit]) ** 2

    print("weight features dict: {}".format(weight_features_dict))
    selected_features = [f for f in features if weight_features_dict[f] > threshold]
    return selected_features
