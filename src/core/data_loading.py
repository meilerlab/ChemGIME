"""
ChemGIME Data Loading
=====================
Functions for loading foundational data files (cofactor sets, metabolite maps,
and generic CSV/TSV data). These are pure data-loading functions with no UI
dependencies.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


def load_cofactor_set(cofactor_file: Path) -> Set[str]:
    """
    Loads a set of canonical cofactor SMILES from a flat file.
    This replaces the 'MIN_SUBSTRATE_ATOM_COUNT' heuristic.
    """
    logger.info(f"Loading cofactor exclusion list from {cofactor_file}...")
    if not cofactor_file.exists():
        logger.warning(f"Cofactor file not found at {cofactor_file}. Proceeding with an empty filter.")
        return set()

    try:
        with open(cofactor_file, 'r') as f:
            cofactors = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(cofactors)} cofactors into the exclusion set.")
        return cofactors
    except Exception as e:
        logger.error(f"Error loading cofactor file {cofactor_file}: {e}")
        return set()


def load_bigg_to_smiles_map(map_file: Path) -> Dict[str, str]:
    """
    Loads the pre-computed TSV map of BiGG metabolite IDs to SMILES.
    This map is essential for parsing the GEM.

    Raises:
        FileNotFoundError: If the map file does not exist.
        ValueError: If the file is missing required columns or contains no data.
    """
    logger.info(f"Loading BiGG Metabolite-to-SMILES map from {map_file}...")
    if not map_file.exists():
        raise FileNotFoundError(
            f"BiGG metabolite map not found at {map_file}. "
            "This file is required to parse GEMs."
        )

    try:
        df = pd.read_csv(map_file, sep='\t')

        if 'bigg_id' not in df.columns or 'smiles' not in df.columns:
            raise ValueError(
                f"Metabolite map {map_file} is missing 'bigg_id' or 'smiles' column."
            )

        df = df.dropna(subset=['bigg_id', 'smiles'])
        df = df.drop_duplicates(subset=['bigg_id'], keep='first')

        smiles_map = df.set_index('bigg_id')['smiles'].to_dict()

        if not smiles_map:
            logger.warning(f"No valid metabolite-to-SMILES mappings were loaded from {map_file}.")
            return {}

        logger.info(f"Loaded {len(smiles_map)} metabolite SMILES mappings.")
        return smiles_map
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading SMILES map {map_file}: {e}") from e


def load_rxn_smiles_map(
    map_file,
) -> Dict[str, tuple]:
    """Load a reaction-level SMILES lookup into a dict.

    The file must be a TSV with at minimum three columns::

        rxn          left_smiles       right_smiles
        rxn00001     CC(=O)O.O         CC(=O)OCC
        ...

    *map_file* can be a ``Path`` (pre-installed database file) or a
    Streamlit ``UploadedFile`` (user-supplied lookup).

    Returns
    -------
    dict
        Mapping of ``rxn_id -> (left_smiles, right_smiles)``.
        Empty dict on failure (never raises, to keep the UI path clean).
    """
    try:
        path_name = getattr(map_file, "name", str(map_file))

        if isinstance(map_file, Path):
            if not map_file.exists():
                logger.info(
                    f"Reaction-SMILES map not found at {map_file}. "
                    "Returning empty map — .tbl-only mode requires this file."
                )
                return {}
            df = pd.read_csv(map_file, sep="\t", dtype=str)
        else:
            # File-like object (UploadedFile from Streamlit)
            first_line = map_file.readline()
            if isinstance(first_line, bytes):
                first_line = first_line.decode("utf-8")
            map_file.seek(0)
            sep = "\t" if "\t" in first_line else ","
            df = pd.read_csv(map_file, sep=sep, dtype=str)

        required = {"rxn", "left_smiles", "right_smiles"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            logger.error(
                f"Reaction-SMILES map {path_name} is missing columns: {missing}. "
                "Expected columns: rxn, left_smiles, right_smiles."
            )
            return {}

        df = df.dropna(subset=["rxn", "left_smiles", "right_smiles"])
        df = df.drop_duplicates(subset=["rxn"], keep="first")
        result = {
            row["rxn"]: (row["left_smiles"], row["right_smiles"])
            for _, row in df.iterrows()
        }
        logger.info(f"Loaded {len(result)} reaction-SMILES mappings from {path_name}.")
        return result

    except Exception as e:
        logger.error(f"Error loading reaction-SMILES map: {e}")
        return {}


def load_data(file_or_path, **kwargs) -> Optional[pd.DataFrame]:
    """Loads data from a file path or a file-like object."""
    try:
        path_name = getattr(file_or_path, 'name', 'unknown') if not hasattr(file_or_path, 'name') else file_or_path.name
        
        if isinstance(file_or_path, Path):
            path_name = file_or_path.name
            with open(file_or_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
        else:
            first_line = file_or_path.readline()
            if isinstance(first_line, bytes):
                first_line = first_line.decode('utf-8')
            file_or_path.seek(0)
            
        sep = '\t' if '\t' in first_line else ','

        df = pd.read_csv(file_or_path, sep=sep, on_bad_lines='warn', **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows from {path_name}")
        return df
    except Exception as e:
        logger.error(f"Error loading file '{getattr(file_or_path, 'name', 'N/A')}': {e}")
        return None
