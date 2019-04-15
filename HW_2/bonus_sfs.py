from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')


def score(x_train: DataFrame, y_train: DataFrame, clf):
    return cross_val_score(clf, x_train, y_train, cv=3).mean()


def sfs_algo(x_train: DataFrame, y_train: DataFrame, clf):
    subset_selected_features = []
    all_features = x_train.columns.values.tolist()
    best_total_score = float('-inf')
    for _ in range(len(all_features)):
        best_score = float('-inf')
        best_feature = None
        unselect_features = [f for f in all_features if f not in subset_selected_features]
        for f in unselect_features:
            current_features = subset_selected_features + [f]
            current_score = score(x_train[current_features], y_train, clf)
            if current_score > best_score:
                best_score = current_score
                best_feature = f
        if best_score > best_total_score:
            best_total_score = best_score
            subset_selected_features.append(best_feature)
        else:
            break
    return subset_selected_features


def run_sfs(x_train: DataFrame, y_train: DataFrame, x_test: DataFrame, y_test: DataFrame, x_val: DataFrame, y_val: DataFrame):
    # examine sfs algorithm with Support Vector Classification
    svr = SVR()
    score_before_sfs = score(x_train, y_train, svr)
    print("Support Vector Classification score before SFS is: {}".format(score_before_sfs))

    selected_features = sfs_algo(x_train, y_train, svr)
    print("Support Vector Classification selected features are: {}".format(selected_features))

    score_after_sfs = score(x_train[selected_features], y_train, svr)
    print("Support Vector Classification score after SFS is: {}".format(score_after_sfs))

    # examine selected features by another classifier which is Decision Tree Classifier
    x_train = x_train[selected_features]
    x_val = x_val[selected_features]
    x_test = x_test[selected_features]

    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(x_train, y_train)
    print("Decision Tree Classifier score after SFS is: {}".format(dtc.score(x_test, y_test)))

    # examine sfs algorithm with K Neighbors Classifier

    knn = KNeighborsClassifier(n_neighbors=5)
    score_before_sfs = score(x_train, y_train, knn)
    print("K Neighbors Classifier score before SFS is: {}".format(score_before_sfs))

    selected_features = sfs_algo(x_train, y_train, knn)
    print("K Neighbors Classifier selected features are: {}".format(selected_features))

    # examine selected features by another classifier which is Decision Tree Classifier
    x_train = x_train[selected_features]
    x_val = x_val[selected_features]
    x_test = x_test[selected_features]

    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(x_train, y_train)
    print("Decision Tree Classifier score after SFS is: {}".format(dtc.score(x_test, y_test)))

    return x_train, y_train, x_val, y_val, x_test, y_test
