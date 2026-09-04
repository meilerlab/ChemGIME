"""
ChemGIME Similarity
===================
Distance calculations and substrate matching functions.
"""

import logging
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def get_closest_rxns(
    query_fp_array: np.ndarray,
    db_fps_array: np.ndarray,
    rxn_table: pd.DataFrame,
    valid_indices: List[int],
    fp_type: str = "drfp",
) -> pd.DataFrame:
    """
    Uses sklearn's pairwise_distances (Jaccard) to find all
    distances in a single matrix operation.

    Note: Tanimoto Distance = Jaccard Distance

    For QRFP fingerprints the quaternary labels are binarised
    (nonzero → 1) before computing Jaccard so that downstream
    consumers (confidence, EC voting) receive the same [0, 1]
    distance they expect.
    """
    if fp_type == "qrfp":
        query_bin = (query_fp_array != 0).astype(np.uint8)
        db_bin = (db_fps_array != 0).astype(np.uint8)
        distances = pairwise_distances(query_bin, db_bin, metric='jaccard')
    else:
        distances = pairwise_distances(
            query_fp_array,
            db_fps_array,
            metric='jaccard'
        )

    tan_distances = distances[0]

    dist_df = pd.DataFrame({
        'original_index': valid_indices,
        'tan_distance': tan_distances
    })

    df_valid = rxn_table.iloc[valid_indices]

    df_rank = pd.merge(
        df_valid.reset_index(names='original_index'),
        dist_df,
        on='original_index'
    )

    return df_rank.sort_values(by="tan_distance").reset_index(drop=True)


def get_all_reactants_from_smiles(
    smiles_string: str,
    cofactor_set: Set[str]
) -> List[Tuple[str, Chem.Mol]]:
    """
    Extracts all valid, non-cofactor molecules from a
    dot-separated SMILES string.
    """
    if not smiles_string or pd.isna(smiles_string):
        return []
    reactants = []
    for smi in smiles_string.split('.'):
        try:
            if not smi or smi in cofactor_set:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol:
                reactants.append((smi, mol))
        except Exception:
            continue
    return reactants


def find_most_similar_reactant(
    query_fp: DataStructs.ExplicitBitVect,
    db_reactants_smiles: str,
    cofactor_set: Set[str]
) -> Tuple[float, str]:
    """
    Finds the most similar reactant in a dot-separated string,
    excluding cofactors.
    """
    if not db_reactants_smiles or pd.isna(db_reactants_smiles) or query_fp is None:
        return 0.0, 'N/A'

    max_similarity, most_similar_smiles = -1.0, 'N/A'

    for smi in db_reactants_smiles.split('.'):
        try:
            if not smi or smi in cofactor_set:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                if (similarity := DataStructs.TanimotoSimilarity(query_fp, fp)) > max_similarity:
                    max_similarity, most_similar_smiles = similarity, smi
        except Exception:
            continue

    return (max_similarity if max_similarity != -1.0 else 0.0), most_similar_smiles
