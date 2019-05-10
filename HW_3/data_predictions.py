from data_infrastructure import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def train_model(x_train: DataFrame, y_train: DataFrame, clf_title: str, clf, k: int):
    """
    :param x_train: train data frame
    :param y_train: label
    :param clf_title title of classifier
    :param clf: classifier
    :param k for K-Cross validation
    :return: fitted classifier and it's score based on K-Cross validation
    """
    kfcv_score = score(x_train, y_train, clf, k)
    fitted_clf = clf.fit(x_train, y_train)
    print(f"{clf_title} {k} - fold cross validation score is: {kfcv_score}")
    return fitted_clf, kfcv_score


def evaluate_fitted_clf(fitted_clf, x_valid: DataFrame, y_valid: DataFrame):
    """
    :param fitted_clf: fitted classifier to evaluate
    :param x_valid: validation data frame
    :param y_valid: validation classes
    :return: scores by metrics
    """
    y_pred = fitted_clf.predict(x_valid)
    acc_score = accuracy_score(y_pred, y_valid)
    return acc_score


def calc_validation_score(clf_title: str, fitted_clf, x_valid: DataFrame, y_valid: DataFrame):
    validation_score = evaluate_fitted_clf(fitted_clf, x_valid, y_valid)
    print(f"{clf_title} validation accuracy score is: {validation_score}")
    return validation_score


def automated_select_classifier(classifer_score_dict: dict) -> str:
    best_clf_score = float('-inf')
    best_clf = None
    for clf_title, clf_data in classifer_score_dict.items():
        clf_score = clf_data[1] + clf_data[2]
        if clf_score > best_clf_score:
            best_clf_score = clf_score
            best_clf = clf_title

    return best_clf


def export_prediction_to_csv(y_test_pred: np.ndarray):
    n2l = np.vectorize(lambda n: num2label[int(n)])
    y_test_pred = n2l(y_test_pred)
    d = {'Vote': y_test_pred}
    DataFrame(d).to_csv("./test_class_predictions.csv", index=False)


def main():
    # Task 1 - load train data frame
    trainDF = load_data(TRAIN_PATH)
    x_train = trainDF[selected_features_without_label]
    y_train = trainDF[label]

    # Task 2 - Train at least 2 models
    classifiers_list = [("RandomForest", RandomForestClassifier(random_state=0,
                                                                criterion='entropy',
                                                                min_samples_split=5,
                                                                min_samples_leaf=3,
                                                                n_estimators=50)),
                        ("MLP", MLPClassifier(random_state=0,
                                              hidden_layer_sizes=(100, 100),
                                              batch_size=32,
                                              learning_rate='adaptive',
                                              max_iter=1000,
                                              activation='relu')),
                        ("SGD", SGDClassifier(random_state=92,
                                              max_iter=1000,
                                              tol=1e-3))
                        ]

    k = 5
    classifiers_fitted_dict = {}
    for clf_title, clf in classifiers_list:
        fitted_clf, kfvc_score = train_model(x_train, y_train, clf_title, clf, k)
        classifiers_fitted_dict[clf_title] = [fitted_clf, kfvc_score]

    # Task 3 - load validation data frame
    validationDF = load_data(VALIDATION_PATH)
    x_valid = validationDF[selected_features_without_label]
    y_valid = validationDF[label]

    # Task 4 - check performance with validation set
    for clf_title, clf_data in classifiers_fitted_dict.items():
        validation_score = calc_validation_score(clf_title, clf_data[0], x_valid, y_valid)
        classifiers_fitted_dict[clf_title].append(validation_score)

    # Task 5 - select the best model for the prediction tasks
    best_clf = automated_select_classifier(classifiers_fitted_dict)
    best_clf_fitted = classifiers_fitted_dict[best_clf][0]

    # Task 6 - Use the selected model to provide predictions
    testDF = load_data(TEST_PATH)
    x_test = testDF[selected_features_without_label]
    y_test = testDF[label]

    # evaluation
    y_test_pred = best_clf_fitted.predict(x_test)

    # confusion matrix
    conf_mat = confusion_matrix(y_target=y_test, y_predicted=y_test_pred, binary=False)
    fig, ax = plot_confusion_matrix(conf_mat=conf_mat)
    plt.show()

    error_score = 1 - accuracy_score(y_test_pred, y_test)
    print(f"The error score is: {error_score}")

    export_prediction_to_csv(y_test_pred)


if __name__ == '__main__':
    main()