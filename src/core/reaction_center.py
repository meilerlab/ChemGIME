"""
ChemGIME Reaction Center Scoring
==================================
Context-aware Tanimoto scoring that weights reaction-center similarity
more heavily than global molecular scaffold similarity.

Problem
-------
Global Tanimoto similarity over entire molecules often misses local
chemical transformations at the enzyme active site.  Two reactions may
share a large molecular scaffold but differ critically at the bond
being broken or formed.

Algorithm
---------
1. Identify the **Reaction Center** — atoms undergoing bond changes —
   using RDKit's reaction SMARTS atom-mapping or Maximum Common
   Substructure (MCS) comparison of reactants vs products.
2. Calculate ``T_global`` — standard Morgan fingerprint Tanimoto over
   the full molecule.
3. Calculate ``T_center`` — Morgan fingerprint Tanimoto over just the
   reaction center substructure.
4. Compute a weighted score:

   .. math::

       FinalScore = \\frac{T_{global} + 2 \\times T_{center}}{3}

   The reaction center similarity is weighted **2×** more heavily than
   the molecular scaffold.

Usage
-----
::

    from src.core.reaction_center import context_aware_tanimoto, batch_context_scores

    score = context_aware_tanimoto(
        query_left="CCO.O",
        query_right="CC=O.O",
        db_left="CCCO.O",
        db_right="CCC=O.O",
    )
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS, rdmolops

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reaction center extraction
# ---------------------------------------------------------------------------

def _get_changed_atoms(
    reactant_mol: Chem.Mol,
    product_mol: Chem.Mol,
    *,
    radius: int = 1,
) -> Optional[Chem.Mol]:
    """Identify atoms that change between reactant and product.

    Strategy: compute the Maximum Common Substructure (MCS) between
    reactant and product.  Atoms NOT in the MCS are the "reaction center"
    on each side.  We then extract the substructure defined by those atoms
    plus a 1-bond neighbourhood (to capture the local chemical context).

    Parameters
    ----------
    reactant_mol, product_mol : Chem.Mol
        The reactant and product molecules.
    radius : int
        Bond radius around changed atoms to include in the center
        fragment.

    Returns
    -------
    Chem.Mol or None
        The reaction center fragment from the reactant side, or None if
        extraction fails.
    """
    if reactant_mol is None or product_mol is None:
        return None

    try:
        # Find the MCS — atoms that are conserved
        mcs_result = rdFMCS.FindMCS(
            [reactant_mol, product_mol],
            timeout=5,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        )

        if not mcs_result.smartsString:
            # No common substructure → entire molecule is the "center"
            return reactant_mol

        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        if mcs_mol is None:
            return reactant_mol

        # Find atoms in reactant that are NOT part of the MCS
        match = reactant_mol.GetSubstructMatch(mcs_mol)
        matched_atoms = set(match) if match else set()
        all_atoms = set(range(reactant_mol.GetNumAtoms()))
        changed_atoms = all_atoms - matched_atoms

        if not changed_atoms:
            # Everything matched → no reaction center detectable
            # Return the full molecule as fallback
            return reactant_mol

        # Expand the center by `radius` bonds
        expanded_atoms = set(changed_atoms)
        for _ in range(radius):
            new_atoms = set()
            for atom_idx in expanded_atoms:
                atom = reactant_mol.GetAtomWithIdx(atom_idx)
                for bond in atom.GetBonds():
                    new_atoms.add(bond.GetBeginAtomIdx())
                    new_atoms.add(bond.GetEndAtomIdx())
            expanded_atoms |= new_atoms

        # Extract the substructure
        edit_mol = Chem.RWMol(reactant_mol)
        atoms_to_remove = sorted(all_atoms - expanded_atoms, reverse=True)
        for idx in atoms_to_remove:
            edit_mol.RemoveAtom(idx)

        try:
            Chem.SanitizeMol(edit_mol)
            return edit_mol.GetMol()
        except Exception:
            # If sanitization fails, return the full molecule
            return reactant_mol

    except Exception as e:
        logger.debug(f"Reaction center extraction failed: {e}")
        return reactant_mol


def extract_reaction_center(
    left_smiles: str,
    right_smiles: str,
    *,
    radius: int = 1,
) -> Optional[Chem.Mol]:
    """Extract the reaction center from dot-separated SMILES.

    For multi-component reactions (A.B >> C.D), we process each
    reactant-product pair and combine the centers.  As a practical
    heuristic, we pick the *largest* reactant and product for MCS
    comparison (since cofactors are typically smaller).

    Parameters
    ----------
    left_smiles : str
        Dot-separated reactant SMILES.
    right_smiles : str
        Dot-separated product SMILES.
    radius : int
        Bond radius around changed atoms.

    Returns
    -------
    Chem.Mol or None
        The reaction center fragment.
    """
    if not left_smiles or not right_smiles:
        return None

    try:
        # Parse all components
        reactant_mols = []
        for smi in left_smiles.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                reactant_mols.append(mol)

        product_mols = []
        for smi in right_smiles.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                product_mols.append(mol)

        if not reactant_mols or not product_mols:
            return None

        # Pick the largest reactant and product by atom count
        main_reactant = max(reactant_mols, key=lambda m: m.GetNumAtoms())
        main_product = max(product_mols, key=lambda m: m.GetNumAtoms())

        return _get_changed_atoms(main_reactant, main_product, radius=radius)

    except Exception as e:
        logger.debug(f"extract_reaction_center failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _morgan_fp(
    mol: Optional[Chem.Mol],
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[DataStructs.ExplicitBitVect]:
    """Compute Morgan fingerprint, returning None on failure."""
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def _tanimoto(
    fp1: Optional[DataStructs.ExplicitBitVect],
    fp2: Optional[DataStructs.ExplicitBitVect],
) -> float:
    """Tanimoto similarity, returning 0.0 if either fingerprint is None."""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ---------------------------------------------------------------------------
# Context-aware scoring
# ---------------------------------------------------------------------------

def context_aware_tanimoto(
    query_left: str,
    query_right: str,
    db_left: str,
    db_right: str,
    *,
    center_weight: float = 2.0,
    global_weight: float = 1.0,
    center_radius: int = 1,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    cofactor_set: Optional[Set[str]] = None,
) -> Tuple[float, float, float]:
    """Compute the context-aware Tanimoto score between two reactions.

    Parameters
    ----------
    query_left, query_right : str
        Dot-separated SMILES for the query reaction.
    db_left, db_right : str
        Dot-separated SMILES for the database reaction.
    center_weight : float
        Weight for T_center (default 2.0).
    global_weight : float
        Weight for T_global (default 1.0).
    center_radius : int
        Bond radius for reaction center extraction.
    fp_radius, fp_bits : int
        Morgan fingerprint parameters.
    cofactor_set : set of str, optional
        Cofactors to exclude before comparison.

    Returns
    -------
    tuple of (final_score, t_global, t_center)
        final_score = (global_weight * T_global + center_weight * T_center) / (global_weight + center_weight)
    """
    def _filter_cofactors(smiles: str) -> str:
        if not cofactor_set or not smiles:
            return smiles
        parts = [s for s in smiles.split(".") if s not in cofactor_set]
        return ".".join(parts)

    q_left = _filter_cofactors(query_left)
    q_right = _filter_cofactors(query_right)
    d_left = _filter_cofactors(db_left)
    d_right = _filter_cofactors(db_right)

    # --- T_global: full-molecule comparison ---
    # Combine all reactant SMILES into one molecule for global comparison
    q_mol = Chem.MolFromSmiles(q_left.replace(".", ".") if q_left else "")
    d_mol = Chem.MolFromSmiles(d_left.replace(".", ".") if d_left else "")

    # For global, use the largest reactant
    q_reactants = [Chem.MolFromSmiles(s) for s in (q_left or "").split(".") if s]
    d_reactants = [Chem.MolFromSmiles(s) for s in (d_left or "").split(".") if s]
    q_main = max(q_reactants, key=lambda m: m.GetNumAtoms()) if q_reactants else None
    d_main = max(d_reactants, key=lambda m: m.GetNumAtoms()) if d_reactants else None

    fp_q_global = _morgan_fp(q_main, fp_radius, fp_bits)
    fp_d_global = _morgan_fp(d_main, fp_radius, fp_bits)
    t_global = _tanimoto(fp_q_global, fp_d_global)

    # --- T_center: reaction center comparison ---
    q_center = extract_reaction_center(q_left, q_right, radius=center_radius)
    d_center = extract_reaction_center(d_left, d_right, radius=center_radius)

    fp_q_center = _morgan_fp(q_center, fp_radius, fp_bits)
    fp_d_center = _morgan_fp(d_center, fp_radius, fp_bits)
    t_center = _tanimoto(fp_q_center, fp_d_center)

    # --- Weighted combination ---
    total_weight = global_weight + center_weight
    if total_weight == 0:
        final_score = 0.0
    else:
        final_score = (global_weight * t_global + center_weight * t_center) / total_weight

    return final_score, t_global, t_center


def batch_context_scores(
    query_left: str,
    query_right: str,
    db_df,  # pd.DataFrame
    *,
    left_col: str = "raw_left_smiles",
    right_col: str = "raw_right_smiles",
    center_weight: float = 2.0,
    global_weight: float = 1.0,
    cofactor_set: Optional[Set[str]] = None,
) -> list:
    """Compute context-aware scores for a query against all DB reactions.

    Parameters
    ----------
    query_left, query_right : str
        Query reaction SMILES.
    db_df : pd.DataFrame
        Database reactions (must contain ``left_col`` and ``right_col``).
    left_col, right_col : str
        Column names for left/right SMILES in db_df.
    center_weight, global_weight : float
        Weighting parameters.
    cofactor_set : set, optional
        Cofactors to exclude.

    Returns
    -------
    list of dict
        One entry per row: ``{final_score, t_global, t_center}``.
    """
    results = []
    for _, row in db_df.iterrows():
        try:
            score, t_g, t_c = context_aware_tanimoto(
                query_left,
                query_right,
                row.get(left_col, ""),
                row.get(right_col, ""),
                center_weight=center_weight,
                global_weight=global_weight,
                cofactor_set=cofactor_set,
            )
            results.append({
                "context_score": score,
                "t_global": t_g,
                "t_center": t_c,
            })
        except Exception as e:
            logger.debug(f"Context scoring failed for row: {e}")
            results.append({
                "context_score": 0.0,
                "t_global": 0.0,
                "t_center": 0.0,
            })

    return results
