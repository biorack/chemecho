# lightweight library for some cheminformatic training and prediction regimes

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import _tree

from rdkit.Chem import AllChem

import joblib
import json
import os
import re


def _count_selfie_frag(encoded_selfie, frag):
    pattern = re.compile(rf"{re.escape(frag)}(?!\d)")
    matches = pattern.findall(encoded_selfie)
    return len(matches)


def _count_smarts(smiles, smarts):
    mol = AllChem.MolFromSmiles(smiles)
    pattern = AllChem.MolFromSmarts(smarts)
    if mol and pattern:
        matches = mol.GetSubstructMatches(pattern)
        return len(matches) 
    return 0
    

def train_substructure_tree(frag, merged_lib, featurized_spectral_data, workdir, polarity,
                            frag_type='group_selfies',
                            max_depth=3,
                            min_frag_count=1,
                            min_positive_unique=10,
                            save_model=True):
    """
    Train decision tree to predict a given group selfie frag or SMARTS pattern

    Parameters:
        frag: either the group name derived from the group selfies package, or a SMARTS pattern
        merged_lib: ms2 library metadata with a 'smiles' and 'frag_encoded_selfie' columns
        frag_type: the type of frag provided to function
        max_depth: maximum depth of tree. Keep the trees shallow if the intention is to build queries
        save_mode: if True, saves model and report to workdir 
    """
    assert frag_type in ['group_selfies', 'smarts']

    if os.path.isfile(f"{workdir}/models/{polarity}_{frag}_model.pkl"):
        print(f'\nAlready Completed Training for {frag}')
        return

    if frag_type == 'group_selfies':
        merged_lib['frag_count'] = merged_lib.frag_encoded_selfie.apply(lambda x: _count_selfie_frag(x, frag))
    elif frag_type == 'smarts':
        merged_lib['frag_count'] = merged_lib.smiles.apply(lambda x: _count_smarts(x, frag))
    
    frag_present = (merged_lib['frag_count'] >= min_frag_count).tolist()
    unique_pos_structures = merged_lib[frag_present].inchikey_smiles.unique()
    
    if len(unique_pos_structures) < min_positive_unique:
        print(f'\nInsufficient positive samples for {frag}')
        return

    X_train, X_test, y_train, y_test = train_test_split(
        featurized_spectral_data, frag_present, test_size=0.2, random_state=42, stratify=frag_present
    )

    clf = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    report_dict = classification_report(y_test, clf.predict(X_test), output_dict=True)
    report_dict['pos_smiles'] = merged_lib[frag_present].smiles.unique().tolist()

    if save_model:
        joblib.dump(clf, f"{workdir}/models/{polarity}_{frag}_model.joblib")
        with open(f"{workdir}/models/{polarity}_{frag}_report.json", "w") as f:
            json.dump(report_dict, f, indent=4)

    return clf, report_dict


def _is_positive_node(tree, node):
    """Return True if node predicts the positive class."""
    values = tree.value[node][0]
    return values[0] < values[1] if len(values) > 1 else values[0] >= 0.5


def _extract_rules(tree, node, conditions, feature_names, positive_class=1, tolerance=None):
    """Recursively extract rules from the entire tree."""
    rules = []
    if tree.feature[node] == _tree.TREE_UNDEFINED:
        if _is_positive_node(tree, node, positive_class):
            rules.append(list(conditions))
        return rules
    feature = feature_names[tree.feature[node]]
    if feature.endswith('_nl'):
        ion_type = "MS2NL"
        subformula = feature[:-3]
    elif feature.endswith('_peak'):
        ion_type = "MS2PROD"
        subformula = feature[:-5]
    base = f"{ion_type}=formula({subformula})"
    if tolerance is not None:
        base += f":TOLERANCEPPM={tolerance}"
    left_condition = f"{base}:INTENSITYPERCENT={tree.threshold[node]:.3f}:EXCLUDED"
    right_condition = f"{base}:INTENSITYPERCENT={tree.threshold[node]:.3f}"
    rules += _extract_rules(tree, tree.children_left[node],
                           conditions + [left_condition],
                           feature_names, positive_class, tolerance)
    rules += _extract_rules(tree, tree.children_right[node],
                           conditions + [right_condition],
                           feature_names, positive_class, tolerance)
    return rules


def convert_tree_to_massql(decision_tree, feature_names, positive_class=1,
                           data_type="MS2DATA", tolerance=None, polarity=None):
    """Convert an entire decision tree into a MassQL query."""
    tree_ = decision_tree.tree_
    rules = _extract_rules(tree_, 0, [], feature_names, positive_class, tolerance)
    if not rules:
        return "-- No rules found."
    queries = []
    for rule in rules:
        query = f"QUERY scaninfo({data_type}) WHERE " + " AND ".join(rule)
        if polarity is not None:
            query += f" AND POLARITY={polarity}"
        queries.append(query)
    return " ||| ".join(queries)