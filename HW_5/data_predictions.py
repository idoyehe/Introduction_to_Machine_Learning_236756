from data_infrastructure import *
from ClassifiersWrapped import *
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


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
    warpper_confusion_matrix(y_valid, y_pred)
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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
    train_df = import_from_csv(TRAIN_PATH)

    x_train = train_df[selected_features_without_label]
    y_train = train_df[label].astype('int')

    # Task 2 - choose a model

    clf = ClassifiersWrapped()

    # Task 3 - Train a classifier
    print("Results using Validation set")
    clf.fit(x_train, y_train)

    # Task 4 - load validation data frame and check model performance
    validation_df = import_from_csv(VALIDATION_PATH)
    x_valid = validation_df[selected_features_without_label]
    y_valid = validation_df[label].astype('int')

    calc_validation_score("Ensemble Classifiers Wrapped ", clf, x_valid, y_valid)

    predictions(clf, x_valid)

    # Task 6 - training model with all training set

    print("Results using Test set")
    train_df = concat([train_df, validation_df])
    x_train = train_df[selected_features_without_label]
    y_train = train_df[label].astype('int')

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
