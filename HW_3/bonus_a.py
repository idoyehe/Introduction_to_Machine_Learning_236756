# Non-Mandatory Assignments -- Task A
def automated_select_classifier(classifer_score_dict: dict) -> str:
    best_clf_score = float('-inf')
    best_clf = None
    for clf_title, clf_data in classifer_score_dict.items():
        clf_score = clf_data[1] + clf_data[2]
        if clf_score > best_clf_score:
            best_clf_score = clf_score
            best_clf = clf_title

    return best_clf
