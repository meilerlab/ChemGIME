import pytest
import pandas as pd
from rdkit import Chem
from src.core.reaction_center import context_aware_tanimoto, extract_reaction_center, _get_changed_atoms

def test_get_changed_atoms():
    # Ethanol to Acetaldehyde
    r_mol = Chem.MolFromSmiles("CCO")
    p_mol = Chem.MolFromSmiles("CC=O")
    
    # Should find the OH -> O= bond change and return that atom + radius
    center = _get_changed_atoms(r_mol, p_mol, radius=1)
    assert center is not None

def test_extract_reaction_center():
    # A multi component reaction
    # A.B >> C.D
    center = extract_reaction_center("CCO.O", "CC=O.O", radius=1)
    assert center is not None

def test_context_aware_tanimoto_identical():
    score, t_g, t_c = context_aware_tanimoto(
        "CCO", "CC=O",
        "CCO", "CC=O",
        center_weight=2.0,
        global_weight=1.0
    )
    assert score == pytest.approx(1.0)
    assert t_g == pytest.approx(1.0)
    assert t_c == pytest.approx(1.0)

def test_context_aware_tanimoto_diff_scaffold():
    q_left = "CCO"
    q_right = "CC=O"
    d_left = "CCCCO"
    d_right = "CCCC=O"
    
    score, t_global, t_center = context_aware_tanimoto(
        query_left=q_left,
        query_right=q_right,
        db_left=d_left,
        db_right=d_right,
        center_weight=2.0,
        global_weight=1.0,
    )
    
    assert t_center > t_global
