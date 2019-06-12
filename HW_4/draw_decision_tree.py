from data_infrastructure import PATH, selected_features_without_label, label2num
from sklearn.tree import export_graphviz


def draw_decision_tree(tree, label):
    export_graphviz(tree,
                    out_file=f'{PATH}trees_output/{label}_decision_tree.dot',
                    feature_names=selected_features_without_label,
                    class_names=[f"NOT_{label}", label],
                    filled=True,
                    )
