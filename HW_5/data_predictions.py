from data_infrastructure import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt


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
    y_pred = fitted_clf.predict(x_valid)
    validation_score = scoring_function(y_pred, y_valid)
    print(f"{clf_title} Classifier accuracy score on validation set is: {validation_score * 100} %")
    return validation_score


def predictions(clf, x_test: DataFrame):
    """
    Gets the chosen classifier for the winner task, prints the winner, and plots
    the votes predicted histogram
    :param clf: Chosen classifier for the task.
    :param x_test: The test features.
    :return: all required predictions
    """
    y_test_prd: np.ndarray = clf.predict(x_test)
    predictions_to_labels = np.vectorize(lambda n: num2label[int(n)])
    y_test_prd = predictions_to_labels(y_test_prd)
    votes_counter = Counter(y_test_prd)
    pred_winner = votes_counter.most_common()[0][0]
    print(f"The predicted party to win the elections is {pred_winner}")

    voters_division = {}
    for key in label2num.keys():
        voters_division[key] = votes_counter[key]

    print(f"The Voters division is {voters_division}")

    votes_normalization = 100 / len(y_test_prd)

    for key in label2num.keys():
        voters_division[key] = votes_counter[key] * votes_normalization

    print(f"The Voters division in percentages is {voters_division}")

    plt.barh(*zip(*sorted(voters_division.items())))
    plt.title("Votes Division")
    plt.xlabel("Votes Percentage %")
    plt.ylabel("Party")
    plt.show()

    return y_test_prd, pred_winner, voters_division


def export_results_to_csv(predictions_vector: np.ndarray, voters_id_col):
    """
    Straight forward from its name
    :param voters_id_col: voters id column
    :param predictions_vector: The predicted Series of Votes
    :return: None
    """
    d = {voters_id: voters_id_col, label: predictions_vector}
    DataFrame(d).to_csv(EXPORT_TEST_PREDICTIONS, index=False)


def main():
    # Task 1 - load train data frame
    train_df = import_from_csv(TRAIN_PATH)

    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    # Task 2 - choose a model

    clf = RandomForestClassifier(random_state=0, criterion='entropy', min_samples_split=3, min_samples_leaf=1, n_estimators=450)

    # Task 3 - Train a classifier
    print("Results using Validation set")
    clf.fit(x_train, y_train)

    # Task 4 - load validation data frame and check model performance
    validation_df = import_from_csv(VALIDATION_PATH)
    x_valid = validation_df[selected_features_without_label]
    y_valid = validation_df[label]

    calc_validation_score("Random Forest", clf, x_valid, y_valid)

    predictions(clf, x_valid)

    # Task 6 - training model with all training set

    print("Results using Test set")
    train_df = concat([train_df, validation_df])
    x_train = train_df[selected_features_without_label]
    y_train = train_df[label]

    clf.fit(x_train, y_train)

    # Task 7 - predict winner color vot division and each voter
    test_df = import_from_csv(TEST_PATH)
    x_test = test_df[selected_features_without_label]
    voters_id_col = test_df[voters_id]

    y_test_prd, pred_winner, voters_division = predictions(clf, x_test)

    # Task 8 - export results
    export_results_to_csv(y_test_prd, voters_id_col)


if __name__ == '__main__':
    main()
