import os
import pickle

import numpy as np
import scipy.sparse as sp

from msbuddy import assign_subformula
from msbuddy.query import query_neutral_mass
from msbuddy.load import init_db
from msbuddy.utils import form_arr_to_str

from tqdm.auto import tqdm

from .utils import in_notebook

disable_progress_bars = not in_notebook()
shared_data_dict = init_db()


def _get_top_subformula(subformula_results):
    """
    Returns a list of the lowest m/z error formula from BUDDY subformula results.
    """
    if subformula_results is None:
        return []

    subform_lists = [res.subform_list for res in subformula_results]

    top_subform = []
    for subform in subform_lists:
        if len(subform) == 0:
            top_subform.append((None, None))
        else:
            top_subform.append((subform[0].formula, subform[0].mass_error_ppm))

    return top_subform


def _get_top_nl_subformula(nl_subformula_results):
    """
    Returns a list of neutral loss formula with the lowest mass difference.
    """
    if nl_subformula_results is None:
        return []

    top_subform = []
    for subform in nl_subformula_results:
        if len(subform) == 0:
            top_subform.append(None)
        else:
            top_subform.append(form_arr_to_str(subform[0].array))

    return top_subform


def _assign_blur_vectors(subformula_lists, nl_results, norm_intensities):
    """
    Computes blurred subformula and neutral loss feature vectors.
    """
    trunc_subformula_lists = [lst for lst in subformula_lists if len(lst) > 0]
    trunc_nl_results = [nl_results[i] for i in range(len(nl_results)) if len(subformula_lists[i]) > 0]

    blurred_subform_spec = []
    blurred_subform_spec_i = []
    for spec_idx, subform_list in enumerate(trunc_subformula_lists):
        intensity = norm_intensities[spec_idx]
        for subform in subform_list:
            blurred_subform_spec.append(subform.formula)
            blurred_subform_spec_i.append(intensity)

    blurred_nl_spec = []
    blurred_nl_spec_i = []
    for spec_idx, nl_list in enumerate(trunc_nl_results):
        intensity = norm_intensities[spec_idx]
        for nl_subform in nl_list:
            blurred_nl_spec.append(form_arr_to_str(nl_subform.array))
            blurred_nl_spec_i.append(intensity)

    peak_vector = np.array([blurred_subform_spec, blurred_subform_spec_i])
    nl_vector = np.array([blurred_nl_spec, blurred_nl_spec_i])
    return peak_vector, nl_vector


def _assign_top_vectors(top_subformula, top_nl_subformula, norm_intensities):
    """
    Computes top subformula and neutral loss feature vectors.
    """
    trunc_top_formula = [formula for formula in top_subformula if formula[0] is not None]
    trunc_top_nl_results = [top_nl_subformula[i] for i in range(len(top_nl_subformula)) if top_subformula[i][0] is not None]

    top_subform_spec = []
    top_subform_spec_i = []
    for i, subform in enumerate(trunc_top_formula):
        intensity = norm_intensities[i]
        top_subform_spec.append(subform[0])
        top_subform_spec_i.append(intensity)

    top_nl_spec = []
    top_nl_spec_i = []
    for i, nl_subform in enumerate(trunc_top_nl_results):
        intensity = norm_intensities[i]
        top_nl_spec.append(nl_subform)
        top_nl_spec_i.append(intensity)

    peak_vector = np.array([top_subform_spec, top_subform_spec_i])
    nl_vector = np.array([top_nl_spec, top_nl_spec_i])
    return peak_vector, nl_vector


def vectorize_spectrum(spec, precursor_mz, parent_formula, adduct, feature_vector_index_map, 
                       max_ppm_error=5, vector_assignment='blur'):
    nl_masses = precursor_mz - spec[0]
    halogens = any(halo in parent_formula for halo in ['Cl', 'F', 'Br', 'I'])

    subformula = assign_subformula(spec[0], parent_formula, adduct=adduct, ms2_tol=max_ppm_error)
    subformula_lists = [sub.subform_list for sub in subformula]
    nl_results = [
        query_neutral_mass(nl_mass, max_ppm_error, True, halogens, shared_data_dict)
        if nl_mass > 0 else []
        for nl_mass in nl_masses
    ]
    
    subform_notna_mask = np.array([len(lst) > 0 for lst in subformula_lists])
    subform_only_i = spec[1][subform_notna_mask]
    norm_intensities = subform_only_i / subform_only_i.sum()

    if vector_assignment == 'blur':
        peak_vector, nl_vector=  _assign_blur_vectors(subformula_lists, nl_results, norm_intensities)
    elif vector_assignment == 'top':
        top_subformula = _get_top_subformula(subformula)
        top_nl_subformula = _get_top_nl_subformula(nl_results)
        peak_vector, nl_vector = _assign_top_vectors(top_subformula, top_nl_subformula, norm_intensities)
    
    row, col, data = [], [], []
    for subform, intensity in zip(peak_vector[0], peak_vector[1]):
        col_name = f'{subform}_peak'
        if col_name in feature_vector_index_map.keys() and not np.isnan(float(intensity)): 
            col_idx = feature_vector_index_map[col_name]
            row.append(0)
            col.append(col_idx)
            data.append(float(intensity))

    for subform, intensity in zip(nl_vector[0], nl_vector[1]):
        col_name = f'{subform}_nl'
        if col_name in feature_vector_index_map.keys() and not np.isnan(float(intensity)): 
            col_idx = feature_vector_index_map[col_name]
            row.append(0)
            col.append(col_idx)
            data.append(float(intensity))

    featurized_spectrum = sp.coo_matrix((data, (row, col)), shape=(1, len(feature_vector_index_map))).tocsr()
    return featurized_spectrum


def _process_spectrum_row(row, vector_assignment, max_ppm_error):
    """
    Processes a single MS2 library row and returns feature vector for peaks and neutral losses.
    """
    spec = row.spectrum
    parent_form = row.parent_formula
    precursor_mz = row.precursor_mz
    nl_masses = precursor_mz - spec[0]
    halogens = any(halo in parent_form for halo in ['Cl', 'F', 'Br', 'I'])

    try:
        subformula = assign_subformula(spec[0], parent_form, adduct=row.adduct, ms2_tol=max_ppm_error)
    except Exception:
        return None, None
    if subformula is None:
        return None, None

    subformula_lists = [sub.subform_list for sub in subformula]
    nl_results = [
        query_neutral_mass(nl_mass, max_ppm_error, True, halogens, shared_data_dict)
        if nl_mass > 0 else []
        for nl_mass in nl_masses
    ]

    subform_notna_mask = np.array([len(lst) > 0 for lst in subformula_lists])
    subform_only_i = spec[1][subform_notna_mask]
    norm_intensities = subform_only_i / subform_only_i.sum()

    if vector_assignment == 'blur':
        return _assign_blur_vectors(subformula_lists, nl_results, norm_intensities)
    elif vector_assignment == 'top':
        top_subformula = _get_top_subformula(subformula)
        top_nl_subformula = _get_top_nl_subformula(nl_results)
        return _assign_top_vectors(top_subformula, top_nl_subformula, norm_intensities)


def subformula_featurization(spectral_df, vector_assignment='blur', max_ppm_error=10):
    """
    Computes subformula feature vectors for peaks and neutral losses for spectral library dataframe.
    """
    assert vector_assignment in ['blur', 'top']

    peak_subformula_vectors = []
    nl_subformula_vectors = []
    for _, row in tqdm(spectral_df.iterrows(), total=spectral_df.shape[0], unit='row', disable=disable_progress_bars):
        peak_vector, nl_vector = _process_spectrum_row(row, vector_assignment, max_ppm_error)
        peak_subformula_vectors.append(peak_vector)
        nl_subformula_vectors.append(nl_vector)

    return peak_subformula_vectors, nl_subformula_vectors


def _build_sparse_matrix(vectors, all_formulas):
    """
    Builds a CSR sparse matrix from a list of spectrum vectors and the given formulas.
    """
    column_index_map = {formula: idx for idx, formula in enumerate(all_formulas)}

    rows, cols, data = [], [], []
    for row_idx, spectrum in enumerate(vectors):
        if spectrum is None:
            continue
        subformulas, intensities = spectrum[0], spectrum[1]
        for subform, intensity in zip(subformulas, intensities):
            if subform in column_index_map and not np.isnan(float(intensity)):
                col_idx = column_index_map[subform]
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(float(intensity))
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(vectors), len(all_formulas)))

    return matrix.tocsr()


def _build_feature_vector_index_map(all_subformula, all_nl_subformula):
    """
    Creates an index mapping for the feature vector columns.
    """
    subformula_cols = [f"{formula}_peak" for formula in all_subformula.tolist()]
    nl_subformula_cols = [f"{formula}_nl" for formula in all_nl_subformula.tolist()]
    feature_vector_cols = subformula_cols + nl_subformula_cols

    return {col: idx for idx, col in enumerate(feature_vector_cols)}


def build_feature_matrix(peak_vectors, nl_vectors):
    """
    Builds the combined feature matrix from peak and neutral loss vectors.
    """
    all_subformula = np.concatenate([subform_spec[0] for subform_spec in peak_vectors if subform_spec is not None])
    all_subformula = np.unique(all_subformula)

    all_nl_subformula = np.concatenate([subform_spec[0] for subform_spec in nl_vectors if subform_spec is not None])
    all_nl_subformula = np.unique(all_nl_subformula)

    peak_matrix = _build_sparse_matrix(peak_vectors, all_subformula)
    nl_matrix = _build_sparse_matrix(nl_vectors, all_nl_subformula)
    featurized_spectral_data = sp.hstack([peak_matrix, nl_matrix])
    feature_vector_index_map = _build_feature_vector_index_map(all_subformula, all_nl_subformula)

    return featurized_spectral_data, feature_vector_index_map


def feature_reduction(featurized_spectral_data, feature_vector_index_map, min_occurence=6):
    """
    Reduces features by minimum occurence, and optionally, correlation
    """
    feature_presence = featurized_spectral_data > 0.0
    feature_counts = np.array(feature_presence.sum(axis=0)).flatten()

    nonrare_features = feature_counts >= min_occurence
    truncated_index_map_keys = np.array(list(feature_vector_index_map.keys()))[nonrare_features]
    truncated_index_map = {str(feature): i for i, feature in enumerate(truncated_index_map_keys)}
    truncated_spectral_data = featurized_spectral_data[:, nonrare_features]

    return truncated_spectral_data, truncated_index_map


def save_featurized_spectra(featurized_spectral_data, feature_vector_index_map, failed_spectra_idxs, workdir, overwrite=False, polarity='positive'):
    assert polarity in ['positive', 'negative']
    assert featurized_spectral_data.shape[1] == len(feature_vector_index_map)
    
    if not os.path.isfile(f'{workdir}/{polarity}_spectral_data.npz') or overwrite:
        sp.save_npz(f'{workdir}/{polarity}_spectral_data.npz', featurized_spectral_data)
    
    if not os.path.isfile(f'{workdir}/{polarity}_index_map.pkl') or overwrite:
        with open(f'{workdir}/{polarity}_index_map.pkl', 'wb') as f:
            pickle.dump(feature_vector_index_map, f)

    if not os.path.isfile(f'{workdir}/{polarity}_failed_spectra.pkl') or overwrite:
        with open(f'{workdir}/{polarity}_failed_spectra.pkl', 'wb') as f:
            pickle.dump(failed_spectra_idxs, f)


def load_featurized_spectra(workdir, polarity='positive'):
    featurized_spectral_data = sp.load_npz(f'{workdir}/{polarity}_spectral_data.npz')

    with open(f'{workdir}/{polarity}_index_map.pkl', 'rb') as f:
        feature_vector_index_map = pickle.load(f)
        
    with open(f'{workdir}/{polarity}_failed_spectra.pkl', 'rb') as f:
        failed_spectra_idxs = pickle.load(f)

    return featurized_spectral_data, feature_vector_index_map, failed_spectra_idxs

