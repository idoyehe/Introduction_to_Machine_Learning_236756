from data_infrastructure import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from bonus_a import *
from collections import defaultdict
from draw_decision_tree import draw_decision_tree


def train_model(x_train: DataFrame, y_train: DataFrame, clf_title: str, clf, k: int):
    """
    :param x_train: train data frame
    :param y_train: label
    :param clf_title title of classifier
    :param clf: classifier
    :return: fitted classifier and it's score based on K-Cross validation
    """
    kfcv_score = score(x_train, y_train, clf, k)
    fitted_clf = clf.fit(x_train, y_train)
    print(f"{clf_title} {k} - fold cross validation score is: {kfcv_score}")
    return fitted_clf, kfcv_score


def evaluate_fitted_clf(fitted_clf, x_valid: DataFrame, y_valid: DataFrame, scoring_function):
    """
    :param fitted_clf: fitted classifier to evaluate
    :param x_valid: validation data frame
    :param y_valid: validation classes
    :param scoring_function: The scoring function to use in order to evaluate
    the classier score. Should take 2 argument: y_true, y_pred, refer to
    accuracy_score function for documentation
    :return: scores by metrics
    """
    y_pred = fitted_clf.predict(x_valid)
    acc_score = scoring_function(y_pred, y_valid)
    return acc_score


def calc_validation_score(clf_title: str, fitted_clf, x_valid: DataFrame, y_valid: DataFrame, scoring_function=accuracy_score):
    """
    Calculates the given classifer score based on the given scoring function
    :param clf_title: Classifier name
    :param fitted_clf: Classifier object, after training
    :param x_valid: features DataFrame
    :param y_valid: True labels Dataframe
    :param scoring_function: scoring function to base score upon.
    Should take 2 argument: y_true, y_pred, refer to accuracy_score function
    for more documentation
    :return: Validation score
    :rtype: float
    """
    validation_score = evaluate_fitted_clf(fitted_clf, x_valid, y_valid, scoring_function)
    print(f"{clf_title} Classifier accuracy score on validation set is: {validation_score * 100} %")
    return validation_score


def winner_color(clf, x_test: DataFrame):
    """
    Gets the chosen classifier for the winner task, prints the winner, and plots
    the votes predicted histogram
    :param clf: Chosen classifier for the task.
    :param x_test: The test features.
    :return: DataFrame with the probabilities for each sample to vote for each party
    """
    y_test_proba: np.ndarray = np.average(clf.predict_proba(x_test), axis=0)
    pred_winner = np.argmax(y_test_proba)
    print(f"The predicted party to win the elections is {num2label[pred_winner]}")
    plt.plot(y_test_proba)  # arguments are passed to np.histogram
    plt.title("Test Vote Probabilities")
    plt.show()
    return y_test_proba


def export_prediction_to_csv(y_test_pred: np.ndarray):
    """
    Straight forward from its name
    :param y_test_pred: The predicted Series of Votes
    :return: None
    """
    predictions_vector = np.vectorize(lambda n: num2label[int(n)])
    exported_y_test_pred = predictions_vector(y_test_pred)
    d = {'Vote': exported_y_test_pred}
    DataFrame(d).to_csv("./test_class_predictions.csv", index=False)


def transportation_service(clf, x_test: DataFrame):
    """
    Solves the task of transportation_service, gets the samples and their
    features and returns a dictionary containing the people each party should
    send transportation to.
    :param clf: The chosen classifier for the job.
    :param x_test: The Samples Dataframe
    :return: dictionary containing the people each party should send
    transportation to.
    """
    y_test_proba: np.ndarray = clf.predict_proba(x_test)
    transportation_dict = defaultdict(list)
    for voter_index, voter in enumerate(y_test_proba):
        for index_color, proba in enumerate(voter):
            if proba > global_transportation_threshold:
                transportation_dict[num2label[index_color]].append(voter_index)

    return transportation_dict


def k_cross_validation_types(clf_type_list: list, k: int, x_train: DataFrame, y_train: DataFrame):
    """
    Returns the best classifier for each type of classifier in clf_type_list.
    :param clf_type_list: a list of elements in the form of (title, (classifiers tuple))
    each element contains a title and an inner tuple of classifiers of the given
    classifier type to chose from.
    :param k: k for k fold cross validation
    :param x_train: The samples DataFrame
    :param y_train: The labels Series
    :return: A dictionary containing the best classifier of each type.
    """
    classifiers_fitted_dict = {}
    for clf_title, clf_tuple in clf_type_list:
        type_clf_best = float('-inf')
        for clf in clf_tuple:
            fitted_clf, kfvc_score = train_model(x_train, y_train, clf_title, clf, k)
            if kfvc_score > type_clf_best:
                classifiers_fitted_dict[clf_title] = fitted_clf
                type_clf_best = kfvc_score

    return classifiers_fitted_dict


def warpper_confusion_matrix(y_target, y_predicted):
    """
    Wrapper function to plot the confusion matrix
    :param y_target: True labels
    :param y_predicted: Predicted lables
    :return: None
    """
    plot_confusion_matrix(y_true=y_target, y_pred=y_predicted, classes=np.asarray([i for i in label2num.keys()]),
                          title='Confusion Matrix')
    plt.show()


def main():
    # Task 1 - load train data frame
    train_df = load_data(TRAIN_PATH)
    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    # Task 2 - Train at least 2 models
    random_forest_tuple = (
        RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=5,
                               min_samples_leaf=3, n_estimators=50),
        RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=3,
                               min_samples_leaf=1, n_estimators=500),
        RandomForestClassifier(random_state=0, criterion='gini', min_samples_split=3,
                               min_samples_leaf=1, n_estimators=500),
    )
    sgd_tuple = (
        SGDClassifier(random_state=0, max_iter=1000, tol=1e-3),
        SGDClassifier(random_state=0, max_iter=1000, tol=1e-2),
        SGDClassifier(random_state=0, max_iter=1500, tol=1e-4)
    )

    knn_tuple = (
        KNeighborsClassifier(n_neighbors=3, algorithm='auto'),
        KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    )

    tree_tuple = (
        DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=5,
                               min_samples_leaf=3),
        DecisionTreeClassifier(random_state=0, criterion='entropy', min_samples_split=3,
                               min_samples_leaf=1)
    )

    classifiers_list = [
        ("RandomForest", random_forest_tuple),
        ("SGD", sgd_tuple),
        ("KNN", knn_tuple),
        ("DecisionTree", tree_tuple)
    ]
    classifiers_fitted_dict = k_cross_validation_types(classifiers_list, 5, x_train, y_train)

    # Task 3 - load validation data frame
    validation_df = load_data(VALIDATION_PATH)
    x_valid = validation_df[selected_features_without_label]
    y_valid = validation_df[label]

    # Task 4 - check performance with validation set
    for clf_title, fitted_clf in classifiers_fitted_dict.items():
        validation_score = calc_validation_score(clf_title, fitted_clf, x_valid, y_valid)
        classifiers_fitted_dict[clf_title] = (fitted_clf, validation_score)

    # Task 5 - select the best model for the prediction tasks
    best_clf = automated_select_classifier(classifiers_fitted_dict)
    print(f"Chosen Classifier{best_clf}")
    best_clf_fitted = classifiers_fitted_dict[best_clf][0]

    # Task 6 - Use the selected model to provide predictions
    test_df = load_data(TEST_PATH)
    x_test = test_df[selected_features_without_label]
    y_test = test_df[label].astype('int')

    # evaluation
    x_train, y_train = concat([x_train, x_valid]), concat([y_train, y_valid])
    best_clf_fitted = best_clf_fitted.fit(x_train, y_train)
    y_test_pred = best_clf_fitted.predict(x_test).astype('int')

    # confusion matrix
    warpper_confusion_matrix(y_target=y_test, y_predicted=y_test_pred)
    acc_score = accuracy_score(y_test_pred, y_test)
    print(f"The accuracy score on TEST set is: {acc_score * 100}%")
    print(f"The error score on TEST set is: {100 - acc_score * 100}%")
    export_prediction_to_csv(y_test_pred)

    # winner prediction
    winner_color(best_clf_fitted, x_test)

    # color vote division
    hist_dict = {}
    for index, _label in num2label.items():
        per = len(y_test_pred[y_test_pred == index]) / len(y_test_pred)
        hist_dict[_label] = str(per * 100) + '%'

    print(hist_dict)

    plt.hist(y_test_pred, density=True)  # arguments are passed to np.histogram
    plt.title("Test Vote Division Histogram")
    plt.show()

    plt.hist(y_train, density=True)  # arguments are passed to np.histogram
    plt.title("All Training Set Vote Division Histogram")
    plt.show()

    # transportation service
    transportation_dict = transportation_service(best_clf_fitted, x_test)
    print(transportation_dict)

    # draw decision tree
    draw_decision_tree(tree=tree_tuple[1])


if __name__ == '__main__':
    main()
