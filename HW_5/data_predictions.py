from data_infrastructure import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter


def train_model(x_train: DataFrame, y_train: DataFrame, clf_title: str, clf, k: int):
    """
    :param x_train: train data frame
    :param y_train: label
    :param clf_title title of classifier
    :param clf: classifier
    :param k: k - fold cross validation parameter
    :return: fitted classifier and it's score based on K-Cross validation
    """
    kfcv_score = score(x_train, y_train, clf, k)
    fitted_clf = clf.fit(x_train, y_train)  # fit clf on entire train dataset
    print(f"{clf_title} {k} - fold cross validation score is: {kfcv_score * 100}%")
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
    Calculates the given classifier score based on the given scoring function
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
    y_test_prd: np.ndarray = clf.predict(x_test)
    pred_winner = Counter(y_test_prd).most_common()[0][0]
    print(f"The predicted party to win the elections is {num2label[pred_winner]}")
    return y_test_prd, pred_winner


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


def main():
    # Task 1 - load train data frame
    train_df = load_data(TRAIN_PATH)

    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    # Task 2 - choose a model

    rf_classifier = RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=3, min_samples_leaf=1, n_estimators=450)

    # Task 3 - Train a classifier

    rf_classifier.fit(x_train, y_train)

    # Task 4 - load validation data frame and check model performance
    validation_df = load_data(VALIDATION_PATH)
    x_valid = validation_df[selected_features_without_label]
    y_valid = validation_df[label]

    calc_validation_score("Random Forest", rf_classifier, x_valid, y_valid)

    # Task 6 - training model with all training set

    train_df = concat([train_df, validation_df])
    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    rf_classifier.fit(x_train, y_train)

    # Task 7 - predict winner color
    test_df = load_data(TEST_PATH)
    x_test = test_df[selected_features_without_label]
    voters_id_col = test_df[voters_id]

    winner_color(rf_classifier, x_test)


if __name__ == '__main__':
    main()
