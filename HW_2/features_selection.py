from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold, RFE
from sklearn import tree


def apply_mi_wrapper_filter(x_train, x_val, y_train, y_val):
    total = len(x_train.columns)
    clf = tree.DecisionTreeClassifier()
    max_score = 0
    for i in range(1, total):
        select_k_best = SelectKBest(mutual_info_classif, k=i).fit(x_train, y_train)
        indices = select_k_best.get_support(indices=True)
        clf.fit(x_train[x_train.columns[indices]], y_train)
        score = clf.score(x_val[x_val.columns[indices]], y_val)
        if max_score < score:
            max_score = score
            best_indices = indices

    print("chosen features after second mi and classifier filter: {}".format(len(best_indices)))
    return list(x_train.columns[best_indices])


def mi_filter(x, y, k):
    selector = SelectKBest(mutual_info_classif, k=k).fit(x, y)
    return selector.get_support()


def variance_filter(x_train, y_train, threshold):
    selector = VarianceThreshold(threshold=threshold).fit(x_train, y_train)
    indices = selector.get_support(indices=True)
    print("number of features after variance filter: {}".format(selector.get_support().sum()))
    return list(x_train.columns[indices])
