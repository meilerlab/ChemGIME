"""
ChemGIME GEM Parser
===================
Functions for parsing Genome-Scale Metabolic Models (GEMs) via cobrapy,
building reaction SMILES, and filtering cofactors.
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def filter_smiles_string(smiles_str: str, cofactor_set: Set[str]) -> str:
    """
    Takes a dot-separated SMILES string and removes any SMILES
    that are present in the cofactor exclusion set.
    """
    if not smiles_str:
        return ""

    filtered_smiles = [
        smi for smi in smiles_str.split('.')
        if smi and smi not in cofactor_set
    ]

    return ".".join(filtered_smiles)


def build_smiles_from_reaction(
    rxn: 'cobra.Reaction',
    met_to_smiles: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Helper function to build dot-separated SMILES strings for a
    single COBRA reaction.

    Returns:
        (left_smiles, right_smiles)
    """
    left_smiles = []
    right_smiles = []

    for met, stoich in rxn.metabolites.items():
        met_id_no_compartment = met.id.rsplit('_', 1)[0]
        smiles = met_to_smiles.get(met_id_no_compartment)

        if not smiles:
            continue

        if stoich < 0:  # Reactant
            left_smiles.append(smiles)
        elif stoich > 0:  # Product
            right_smiles.append(smiles)

    if not left_smiles or not right_smiles:
        return None, None

    return ".".join(left_smiles), ".".join(right_smiles)


def build_organism_database(
    _gem_file_obj: Any,
    met_to_smiles: Dict[str, str],
    cofactor_set: Set[str]
) -> Optional[pd.DataFrame]:
    """
    Parses a GEM file (SBML/XML) and builds a DataFrame of reactions,
    GPRs, ECs, and filtered SMILES.

    Args:
        _gem_file_obj: A file-like object with .seek(), .read(), and .name.
        met_to_smiles: The BiGG ID to SMILES mapping.
        cofactor_set: The set of cofactor SMILES to exclude.

    Returns:
        DataFrame with reaction data, or None if no valid reactions found.

    Raises:
        RuntimeError: If the SBML file cannot be parsed.
    """
    try:
        import cobra
    except ImportError:
        raise ImportError(
            "`cobra` library not found. Please install it: `pip install cobra`"
        )

    try:
        _gem_file_obj.seek(0)
        file_content = _gem_file_obj.read().decode('utf-8')
        model = cobra.io.read_sbml_model(file_content, use_fbc_package=False)
        logger.info(f"Successfully loaded GEM: {model.id} with {len(model.reactions)} reactions.")
    except Exception as e:
        raise RuntimeError(
            f"Error parsing SBML file {_gem_file_obj.name}: {e}"
        ) from e

    reaction_data = []
    for rxn in model.reactions:
        if rxn.boundary:
            continue

        gpr_string = rxn.gene_reaction_rule
        ec_number = rxn.annotation.get('ec-code', '')
        if isinstance(ec_number, list):
            ec_number = ";".join(ec_number)

        raw_left_smiles, raw_right_smiles = build_smiles_from_reaction(rxn, met_to_smiles)

        if not raw_left_smiles or not raw_right_smiles:
            continue

        clean_left_smiles = filter_smiles_string(raw_left_smiles, cofactor_set)
        clean_right_smiles = filter_smiles_string(raw_right_smiles, cofactor_set)

        if raw_left_smiles and raw_right_smiles:
            reaction_data.append({
                "reaction_id": rxn.id,
                "name": rxn.name,
                "gpr": gpr_string,
                "ec": ec_number,
                "subsystem": rxn.subsystem,
                "raw_left_smiles": clean_left_smiles,
                "raw_right_smiles": clean_right_smiles,
            })

    if not reaction_data:
        logger.warning(f"No valid, non-boundary reactions with SMILES could be extracted from {model.id}.")
        return None

    logger.info(f"Extracted {len(reaction_data)} valid reactions from {model.id}.")
    return pd.DataFrame(reaction_data)
