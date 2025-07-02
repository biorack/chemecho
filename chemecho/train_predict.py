# lightweight library for some cheminformatic training and prediction regimes

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import _tree

from rdkit.Chem import AllChem

import numpy as np
import joblib
import json
import os
import re


def filter_failed_idxs(featurized_spectral_data, merged_lib, failed_spectra_idxs):
    """
    Remove indicies that failed the vectorization process from matrix and dataframe
    """
    filter_mask = np.ones(featurized_spectral_data.shape[0], dtype=bool)
    filter_mask[failed_spectra_idxs] = False
    filtered_spectral_data = featurized_spectral_data[filter_mask]

    failed_df_mask = merged_lib.index.isin(failed_spectra_idxs)
    filtered_merged_lib = merged_lib.iloc[~failed_df_mask].reset_index(drop=True)

    return filtered_spectral_data, filtered_merged_lib


def _get_canonical_smiles(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol:
        return AllChem.MolToSmiles(mol, canonical=True)
    return None


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
                            class_weights='balanced',
                            save_model=True):
    """
    Train decision tree to predict a given group selfie frag or SMARTS pattern

    Parameters:
        frag: either the group name derived from the group selfies package, or a SMARTS pattern
        merged_lib: ms2 library metadata with a 'smiles' and 'frag_encoded_selfie' columns
        featurized_spectral_data: previously embedded library spectra
        workdir: path to working directory
        polarity: either 'negative' or 'positive'
        frag_type: the type of frag provided to function
        max_depth: maximum depth of tree. Keep the trees shallow if the intention is to build queries
        min_frag_count: the minimum number of substructures to be labeled as positive
        min_positive_unique: the minimum number of unique structures in training dataset
        save_mode: if True, saves model and report to workdir 
    """
    assert frag_type in ['group_selfies', 'smarts']

    if not os.path.isdir(f"{workdir}/models/"):
        os.mkdir(f"{workdir}/models/")

    model_path = f"{workdir}/models/{polarity}_{frag}_model.joblib"
    report_path = f"{workdir}/models/{polarity}_{frag}_report.json"

    if os.path.isfile(model_path):
        clf = joblib.load(model_path)
        with open(report_path, "r") as f:
            report_dict = json.load(f)
        return clf, report_dict

    if frag_type == 'group_selfies':
        merged_lib['frag_count'] = merged_lib.frag_encoded_selfie.apply(lambda x: _count_selfie_frag(x, frag))
    elif frag_type == 'smarts':
        merged_lib['frag_count'] = merged_lib.smiles.apply(lambda x: _count_smarts(x, frag))
    
    merged_lib['frag_present'] = (merged_lib['frag_count'] >= min_frag_count).tolist()
    merged_lib['canonical_smiles'] = merged_lib['smiles'].apply(_get_canonical_smiles)
    smiles_grouped = merged_lib.groupby('canonical_smiles')['frag_present'].agg(lambda x: int(x.max())).reset_index()
    
    unique_pos_structures = merged_lib[merged_lib['frag_present']].inchikey_smiles.unique()
    
    if len(unique_pos_structures) < min_positive_unique:
        print(f'\nInsufficient positive samples for {frag}')
        return

    # split by canonical smiles rather than spectra
    train_groups, test_groups = train_test_split(
        smiles_grouped,
        test_size=0.2,
        stratify=grouped['frag_present'],
        random_state=42
    )

    merged_lib = merged_lib.reset_index(drop=True)
    merged_lib['row_index'] = merged_lib.index
    
    train_indices = merged_lib[merged_lib['canonical_smiles'].isin(train_groups['canonical_smiles'])]['row_index'].values
    test_indices  = merged_lib[merged_lib['canonical_smiles'].isin(test_groups['canonical_smiles'])]['row_index'].values
    
    X_train = featurized_spectral_data[train_indices]
    X_test  = featurized_spectral_data[test_indices]
    
    frag_present = merged_lib['frag_present'].to_numpy()
    y_train = frag_present[train_indices]
    y_test  = frag_present[test_indices]

    clf = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, class_weight=class_weights)
    clf.fit(X_train, y_train)
    
    report_dict = classification_report(y_test, clf.predict(X_test), output_dict=True)
    report_dict['pos_smiles'] = merged_lib[frag_present].smiles.unique().tolist()

    if save_model:
        joblib.dump(clf, model_path)
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

    return clf, report_dict


def _is_positive_node(tree, node):
    """Return True if node predicts the positive class."""
    values = tree.value[node][0]
    return values[0] < values[1] if len(values) > 1 else values[0] >= 0.5


def _extract_rules(tree, node, conditions, feature_names, precision, positive_class, tolerance):
    """Recursively extract rules from the entire tree."""
    rules = []
    if tree.feature[node] == _tree.TREE_UNDEFINED:
        if _is_positive_node(tree, node):
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
    left_condition = f"{base}:INTENSITYPERCENT={tree.threshold[node]:.{precision}f}:EXCLUDED"
    right_condition = f"{base}:INTENSITYPERCENT={tree.threshold[node]:.{precision}f}"
    rules += _extract_rules(tree, tree.children_left[node],
                           conditions + [left_condition],
                           feature_names, precision, positive_class, tolerance)
    rules += _extract_rules(tree, tree.children_right[node],
                           conditions + [right_condition],
                           feature_names, precision, positive_class, tolerance)
    return rules


def convert_tree_to_massql(decision_tree, feature_names, positive_class=1,
                           data_type="MS2DATA", tolerance=None, polarity=None, precision=3):
    """Convert an entire decision tree into a MassQL query."""
    tree_ = decision_tree.tree_
    rules = _extract_rules(tree_, 0, [], feature_names, precision, positive_class, tolerance)
    if not rules:
        return "-- No rules found."
    queries = []
    for rule in rules:
        query = f"QUERY scaninfo({data_type}) WHERE " + " AND ".join(rule)
        if polarity is not None:
            query += f" AND POLARITY={polarity}"
        queries.append(query)
    return " ||| ".join(queries)