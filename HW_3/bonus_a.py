# Non-Mandatory Assignments -- Task A
def automated_select_classifier(classifier_score_dict: dict) -> str:
    """
    Gets a dictionary of classifiers: {classifier_title: classifier}
    :param classifier_score_dict:
    :return:
    """
    best_clf_score = float('-inf')
    best_clf = None
    for clf_title, clf_data in classifier_score_dict.items():
        clf_score = clf_data[1]
        if clf_score > best_clf_score:
            best_clf_score = clf_score
            best_clf = clf_title

    return best_clf
