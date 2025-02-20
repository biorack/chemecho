import logging
from IPython.display import display, HTML

import os
import sys
import requests
import numpy as np
import pandas as pd
from io import StringIO

from rdkit import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from tqdm.auto import tqdm
tqdm.pandas()

pd.options.mode.chained_assignment = None  # default='warn'


###########
# Logging #
###########


class JupyterNotebookHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        display(HTML(f"<pre>{log_entry}</pre>"))
        sys.stdout.flush()  # Force immediate output


def in_notebook():
    try:
        from IPython import get_ipython
        if 'ZMQInteractiveShell' in str(type(get_ipython())):
            return True
        else:
            return False
    except:
        return False


def setup_logger(name="logger", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if in_notebook():
        jupyter_handler = JupyterNotebookHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        jupyter_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(jupyter_handler)
        logger.info("Jupyter notebook detected, using notebook logging.")
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        logger.info("Command-line environment detected, using generic logging.")

    return logger

logger = setup_logger()
disable_progress_bars = not in_notebook()


####################################
# Download & Process MS2 Libraries #
####################################

ALLOWABLE_ADDUCTS = ['[M-H]-', '[M-H]1-', '[2M-H]-', '[2M-H]1-', '[M+Cl]-', '[M+Cl]1-', '[M-H-H2O]-', '[M-H-H2O]1-', '[M-2H]2-', '[M+HCOOH-H]1-', '[M-H-CO2]-',
                     '[M+H]+', '[M+H]1+', '[2M+H]+', '[2M+H]1+', '[M+Na]+', '[M+Na]1+', '[M+H-H2O]+', '[M+H-H2O]1+', '[M-H2O+H]1+', '[M+2H]2+', '[M+NH4]+', '[M+K]1+', '[M+NH3+H]1+']


def _peaks_to_spectrum(peaks_json):
    """
    Convert JSON peak lists into numpy array format.
    """
    try:
        peaks = eval(peaks_json)
        mzs, intensities = zip(*peaks)

        return np.array([mzs, intensities])
    except:
        return None


def _download_processed_gnps_data():
    """
    Downloads GNPS cleaned library and loads it as a pandas DataFrame.
    """
    url = 'https://external.gnps2.org/processed_gnps_data/gnps_cleaned.json'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_json(data)
        return df
    else:
        raise Exception(f"Failed to download the file. Status code: {response.status_code}")


def load_processed_gnps_data(gnps_cleaned_path='./data/gnps_cleaned.tsv', min_ppm_diff=5, convert_spectra=True, polarity='positive'):
    """
    Loads processed GNPS spectral libraries.
    """
    if os.path.isfile(gnps_cleaned_path):
        logger.info(f"Cleaned GNPS lib detected, not downloading.")
        gnps_cleaned = pd.read_csv(gnps_cleaned_path, sep='\t', index_col=0, 
                               dtype={'scan': int, 'spectrum_id': "string", 'collision_energy': "string", 'Adduct': "string",
                                      'Compound_Source': "string", 'Compound_Name':"string", 'Precursor_MZ': float, 'ExactMass': float,
                                      'Charge': int, 'Ion_Mode': "string", 'Smiles': "string", 'INCHI': "string", 'InChIKey_smiles': "string",
                                      'msManufacturer': "string", 'msMassAnalyzer': "string", 'msIonisation': "string", 'msDissociationMethod': "string",
                                      'GNPS_library_membership': "string", 'ppmBetweenExpAndThMass': float, 'peaks_json': "string"})
    else:
        logger.info(f"Cleaned GNPS lib not detected, downloading.")
        gnps_cleaned = _download_processed_gnps_data()
        gnps_cleaned.to_csv(gnps_cleaned_path, sep='\t')
        
    gnps_cleaned = gnps_cleaned[gnps_cleaned['ppmBetweenExpAndThMass'] < min_ppm_diff]
    gnps_cleaned.columns = [col.lower() for col in gnps_cleaned.columns]

    gnps_cleaned = gnps_cleaned[gnps_cleaned.ion_mode == polarity]
    gnps_cleaned = gnps_cleaned[gnps_cleaned.adduct.isin(ALLOWABLE_ADDUCTS)].reset_index(drop=True)

    if convert_spectra:
        logger.info(f"Converting GNPS json peaks to numpy spectra.")
        gnps_cleaned['spectrum'] = gnps_cleaned['peaks_json'].progress_apply(_peaks_to_spectrum)
        gnps_cleaned = gnps_cleaned[pd.notna(gnps_cleaned['spectrum'])]
    
    return gnps_cleaned


def merge_in_nist(gnps_cleaned, nist_cleaned_path='./data/nist_cleaned.tsv', convert_spectra=True, polarity='positive'):
    """
    Merge GNPS and NIST cleaned datasets
    """
    assert os.path.isfile(nist_cleaned_path), "Invalid path to nist library"
    
    nist_cleaned = pd.read_csv(nist_cleaned_path, sep='\t')
    nist_cleaned = nist_cleaned[nist_cleaned.ion_mode == polarity]
    nist_cleaned = nist_cleaned[nist_cleaned.adduct.isin(ALLOWABLE_ADDUCTS)]
    
    if convert_spectra:
        logger.info(f"Converting NIST json peaks to numpy spectra.")
        nist_cleaned['spectrum'] = nist_cleaned['spectrum'].progress_apply(lambda x: np.array(eval(x)))
    
    merged_lib = pd.concat([gnps_cleaned, nist_cleaned]).reset_index(drop=True)
    merged_lib['parent_formula'] = merged_lib.smiles.apply(lambda x: CalcMolFormula(AllChem.MolFromSmiles(x)))
    return merged_lib