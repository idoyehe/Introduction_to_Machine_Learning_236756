from data_infrastructure import PATH, selected_features_without_label, label2num
from sklearn.tree import export_graphviz


def draw_decision_tree(tree):
    export_graphviz(tree,
                    out_file=PATH + 'decision_tree.dot',
                    feature_names=selected_features_without_label,
                    class_names=list(label2num.keys()),
                    filled=True,
                    )
