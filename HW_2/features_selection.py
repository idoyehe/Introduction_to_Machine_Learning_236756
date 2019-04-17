from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold, RFE
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


def apply_mi_wrapper_filter(x_train, y_train):
    total = int(len(x_train.columns) / 2)
    clf = SGDClassifier(random_state=92, max_iter=1000, tol=1e-3)
    max_score = 0
    best_indices = []
    for i in range(1, total + 1):
        select_k_best = SelectKBest(mutual_info_classif, k=i).fit(x_train, y_train)
        indices = select_k_best.get_support(indices=True)
        score = cross_val_score(clf, x_train[x_train.columns[indices]], y_train, cv=3, scoring='accuracy').mean()
        print("k is: {} and score is: {}".format(i, score))
        if score > max_score:
            max_score = score
            best_indices = indices

    print("chosen features after SelectKBest and classifier filter: {}".format(len(best_indices)))
    return list(x_train.columns[best_indices])


def mi_filter(x, y, k):
    selector = SelectKBest(mutual_info_classif, k=k).fit(x, y)
    return selector.get_support()


def variance_filter(x_train, y_train, variance_threshold):
    selector = VarianceThreshold(threshold=variance_threshold).fit(x_train, y_train)
    indices = selector.get_support(indices=True)
    print("number of features after variance filter: {}".format(selector.get_support().sum()))
    return list(x_train.columns[indices])
