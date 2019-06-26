from ClassifiersWrapped import *
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def calc_accuracy_score(clf_title: str, fitted_clf, x_valid: DataFrame, y_valid: DataFrame, set_name: str):
    y_pred_forest = fitted_clf.clf1_predict(x_valid)
    print(f"Accuracy with validation set score of Random Forest: {accuracy_score(y_true=y_valid, y_pred=y_pred_forest) * 100} %")
    warpper_confusion_matrix(y_predicted=y_pred_forest, y_target=y_valid)

    y_pred_mlp = fitted_clf.clf2_predict(x_valid)
    print(f"Accuracy with validation set score of MLP: {accuracy_score(y_true=y_valid, y_pred=y_pred_mlp) * 100} %")
    warpper_confusion_matrix(y_predicted=y_pred_mlp, y_target=y_valid)

    y_pred_svc = fitted_clf.clf2_predict(x_valid)
    print(f"Accuracy with validation set score of SVC: {accuracy_score(y_true=y_valid, y_pred=y_pred_svc) * 100} %")
    warpper_confusion_matrix(y_predicted=y_pred_svc, y_target=y_valid)

    y_pred_ensamble = fitted_clf.predict(x_valid)
    validation_score = accuracy_score(y_pred=y_pred_ensamble, y_true=y_valid)
    print(f"{clf_title} Classifier accuracy score on {set_name} set is: {validation_score * 100} %")
    warpper_confusion_matrix(y_valid, y_pred_ensamble)
    return validation_score


def predictions(clf, x_test: DataFrame, y_test: Series = None):
    y_test_prd: np.ndarray = clf.predict(x_test)

    if y_test:
        print(f"Accuracy with test set score of classifier: {accuracy_score(y_true=y_test, y_pred=y_test_prd) * 100} %")
        print(f"The actual party won the elections is {num2label[int(Counter(y_test).most_common()[0][0])]}")

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
    # Task 1 - loading train data frame
    train_df = import_from_csv(TRAIN_PATH)

    x_train = train_df[selected_features_without_label]
    y_train = train_df[label].astype('int')

    # Task 2 - K Cross Validating on the training set

    clf = ClassifiersWrapped(4, 100)
    score_clf1 = clf.classifier1_k_cross_valdition(x_train, y_train)
    print(f"Accuracy K cross validation score of Random Forest: {score_clf1}")

    score_clf2 = clf.classifier2_k_cross_valdition(x_train, y_train)
    print(f"Accuracy K cross validation score of MLP: {score_clf2}")

    score_clf3 = clf.classifier3_k_cross_valdition(x_train, y_train)
    print(f"Accuracy K cross validation score of SVC: {score_clf3}")

    # Task 3 - checking performance on validation set

    print("Results each classifier using Validation set")

    validation_df = import_from_csv(VALIDATION_PATH)
    x_valid = validation_df[selected_features_without_label]
    y_valid = validation_df[label].astype('int')

    calc_accuracy_score("Ensemble Classifiers Wrapped ", clf, x_valid, y_valid, "validation")

    # Task 5 - check performance with unseen test set
    print("Results labeled Test set")
    test_labeled_df = import_from_csv(TEST_PATH)
    x_test_labeled = test_labeled_df[selected_features_without_label]
    y_test_labeled = test_labeled_df[label].astype('int')

    predictions(clf, x_test_labeled, y_test_labeled)

    # Task 6 - training model with all training set

    print("Results using unlabeled Test set")
    train_df = concat([train_df, validation_df, test_labeled_df])

    x_train = train_df[selected_features_without_label]
    y_train = train_df[label].astype('int')

    clf.fit(x_train, y_train)

    # Task 7 - predict winner color vote division and each voter
    test_df = import_from_csv(TEST_UNLABELED_PATH)
    x_test = test_df[selected_features_without_label]
    voters_id_col = test_df[voters_id]

    y_test_prd, pred_winner, voters_division = predictions(clf, x_test)

    # Task 8 - export results
    export_results_to_csv(y_test_prd, voters_id_col)


if __name__ == '__main__':
    main()
