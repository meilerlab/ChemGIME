"""Unit tests for the specificity / decoy-discrimination metrics.

Cases use small inputs with analytically known answers so a regression in
the metric layer is caught independently of the benchmark pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.specificity import (
    average_precision,
    bedroc,
    bootstrap_ci,
    enrichment_factor,
    precision_at_k,
    rates_at_threshold,
    recall_at_k,
    roc_auc,
    roc_points,
)
from benchmarks.decoy_generation import (
    ec_matches,
    label_candidates,
    make_scrambled_reactions,
)


# ── ROC-AUC ────────────────────────────────────────────────────────────────

def test_roc_auc_perfect_separation():
    # Actives all score higher than decoys -> AUC = 1.
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.4, 0.1]
    assert roc_auc(labels, scores) == pytest.approx(1.0)


def test_roc_auc_perfect_inversion():
    labels = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_roc_auc_single_class_returns_none():
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None


def test_roc_auc_known_value():
    # 1 active (0.6), decoys at 0.9/0.5/0.1 -> active beats 2 of 3 -> AUC=2/3.
    labels = [1, 0, 0, 0]
    scores = [0.6, 0.9, 0.5, 0.1]
    assert roc_auc(labels, scores) == pytest.approx(2 / 3)


def test_roc_points_shape():
    fpr, tpr, thr = roc_points([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert fpr[0] == 0.0 and tpr[-1] == 1.0
    assert len(fpr) == len(tpr) == len(thr)


# ── Precision / recall @ k ──────────────────────────────────────────────────

def test_precision_at_k():
    labels = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]  # top-2: active, decoy
    assert precision_at_k(labels, scores, 1) == pytest.approx(1.0)
    assert precision_at_k(labels, scores, 2) == pytest.approx(0.5)
    assert precision_at_k(labels, scores, 3) == pytest.approx(2 / 3)


def test_precision_at_k_clamps():
    assert precision_at_k([1, 0], [0.9, 0.1], 10) == pytest.approx(0.5)


def test_recall_at_k():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6]  # 2 actives total; top-3 contains both
    assert recall_at_k(labels, scores, 3) == pytest.approx(1.0)
    assert recall_at_k(labels, scores, 1) == pytest.approx(0.5)


# ── Enrichment factor ───────────────────────────────────────────────────────

def test_enrichment_factor_max():
    # 100 candidates, 1 active ranked first; EF@1% = (1/1)/(1/100) = 100.
    labels = [0] * 100
    labels[0] = 1
    scores = list(np.linspace(1.0, 0.0, 100))
    assert enrichment_factor(labels, scores, fraction=0.01) == pytest.approx(100.0)


def test_enrichment_factor_random_is_one():
    # Active ratio equals top fraction and active sits exactly at the cut.
    labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = list(np.linspace(1, 0, 10))
    # top 10% = 1 candidate, the active -> EF = (1/1)/(1/10) = 10 (max).
    assert enrichment_factor(labels, scores, fraction=0.1) == pytest.approx(10.0)


def test_enrichment_factor_no_actives_none():
    assert enrichment_factor([0, 0, 0], [0.1, 0.2, 0.3]) is None


# ── BEDROC ───────────────────────────────────────────────────────────────────

def test_bedroc_bounds_and_ordering():
    n = 50
    labels = [0] * n
    # Actives at the very top -> BEDROC near 1.
    for i in range(5):
        labels[i] = 1
    scores = list(np.linspace(1, 0, n))
    good = bedroc(labels, scores, alpha=20.0)
    # Actives at the very bottom -> BEDROC near 0.
    labels_bad = [0] * n
    for i in range(5):
        labels_bad[-(i + 1)] = 1
    bad = bedroc(labels_bad, scores, alpha=20.0)
    # RDKit's BEDROC can land a hair outside [0,1] from floating point.
    assert -1e-9 <= bad < good <= 1.0 + 1e-9
    assert good > 0.9
    assert bad < 0.1


def test_bedroc_single_class_none():
    assert bedroc([1, 1, 1], [0.1, 0.2, 0.3]) is None


# ── Rates at threshold ───────────────────────────────────────────────────────

def test_rates_at_threshold():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.6, 0.1]  # threshold 0.5 accepts idx 0 (TP) and 2 (FP)
    r = rates_at_threshold(labels, scores, 0.5)
    assert r["tp"] == 1 and r["fn"] == 1
    assert r["fp"] == 1 and r["tn"] == 1
    assert r["tpr"] == pytest.approx(0.5)
    assert r["fpr"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(0.5)


# ── Average precision ────────────────────────────────────────────────────────

def test_average_precision_perfect():
    assert average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def test_bootstrap_ci_contains_mean():
    vals = [0.8, 0.82, 0.79, 0.85, 0.81]
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000, seed=0)
    assert lo <= mean <= hi
    assert mean == pytest.approx(np.mean(vals), abs=1e-9)


def test_bootstrap_ci_drops_nan():
    mean, lo, hi = bootstrap_ci([1.0, np.nan, 1.0], n_boot=500, seed=0)
    assert mean == pytest.approx(1.0)


# ── Input validation ─────────────────────────────────────────────────────────

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        roc_auc([1, 0], [0.5])


def test_non_binary_labels_raise():
    with pytest.raises(ValueError):
        roc_auc([1, 2, 0], [0.1, 0.2, 0.3])


# ── EC matching ──────────────────────────────────────────────────────────────

def test_ec_matches_levels():
    assert ec_matches("1.1.1.42", "1.1.1.42", level=4)
    assert ec_matches("1.1.1.42", "1.1.1.99", level=3)
    assert not ec_matches("1.1.1.42", "1.1.1.99", level=4)
    assert not ec_matches("1.1.1.42", "2.1.1.42", level=1)
    assert not ec_matches("1.1.1.42", "1.1.-.-", level=3)  # wildcard never matches
    assert not ec_matches("", "1.1.1.1", level=1)


# ── Cross-EC labelling ───────────────────────────────────────────────────────

def _ranked(rows):
    return pd.DataFrame(rows)


def test_label_candidates_basic():
    # Active = R_active (explicit target). R_decoy is similar + different EC.
    # R_same is similar + same EC class (ambiguous). R_easy is dissimilar.
    ranked = _ranked([
        {"reaction_id": "R_active", "ec": "1.1.1.42", "tan_distance": 0.10},
        {"reaction_id": "R_decoy",  "ec": "2.7.1.1",  "tan_distance": 0.30},
        {"reaction_id": "R_same",   "ec": "1.1.1.99", "tan_distance": 0.35},
        {"reaction_id": "R_easy",   "ec": "3.1.1.1",  "tan_distance": 0.80},
    ])
    lab = label_candidates(ranked, ["R_active"], target_ec="1.1.1.42",
                           decoy_level=3, sim_threshold=0.5)
    assert lab.n_active == 1
    assert lab.n_decoy == 1          # hard (in-window) decoys
    assert lab.n_decoy_all == 2      # R_decoy + R_easy (both different class)
    assert lab.n_ambiguous == 1
    assert list(lab.category) == ["active", "decoy_hard", "ambiguous", "decoy_far"]

    hard_labels, hard_scores = lab.hard_subset()
    assert hard_labels.tolist() == [1, 0]          # active first, decoy second
    assert hard_scores[0] == pytest.approx(0.90)   # 1 - 0.10
    assert hard_scores[1] == pytest.approx(0.70)   # 1 - 0.30


def test_label_candidates_excludes_self():
    ranked = _ranked([
        {"reaction_id": "SELF", "ec": "1.1.1.42", "tan_distance": 0.0},
        {"reaction_id": "R2",   "ec": "2.1.1.1",  "tan_distance": 0.2},
    ])
    lab = label_candidates(ranked, ["SELF", "R2"], target_ec="1.1.1.42",
                           exclude_rxn_ids=["SELF"])
    assert lab.category[0] == "excluded"
    assert lab.n_active == 1  # only R2 remains an active


def test_label_candidates_decoy_requires_similarity():
    # Different EC but far away -> easy negative, not a decoy.
    ranked = _ranked([
        {"reaction_id": "R_active", "ec": "1.1.1.42", "tan_distance": 0.1},
        {"reaction_id": "R_far",    "ec": "2.7.1.1",  "tan_distance": 0.9},
    ])
    lab = label_candidates(ranked, ["R_active"], target_ec="1.1.1.42",
                           sim_threshold=0.5)
    assert lab.n_decoy == 0          # no hard (in-window) decoy
    assert lab.n_decoy_all == 1      # still a global decoy
    assert lab.category[1] == "decoy_far"


def test_label_candidates_same_class_is_ambiguous_not_decoy():
    # A similar reaction sharing the query EC class is an isozyme-like hit,
    # not a false positive: it must not be counted as a decoy.
    ranked = _ranked([
        {"reaction_id": "R_active", "ec": "1.1.1.42", "tan_distance": 0.1},
        {"reaction_id": "R_iso",    "ec": "1.1.1.42", "tan_distance": 0.2},
    ])
    lab = label_candidates(ranked, ["R_active"], target_ec="1.1.1.42",
                           decoy_level=3, sim_threshold=0.5)
    assert lab.n_decoy == 0
    assert lab.n_ambiguous == 1


# ── Scrambled-reaction decoys ─────────────────────────────────────────────────

def test_make_scrambled_reactions_basic():
    decoys = make_scrambled_reactions(
        "CCO", "CC=O", ["CC(=O)O", "C1CCCCC1", "CCN"], n_decoys=10, seed=0)
    assert len(decoys) == 3
    # Every decoy keeps the true reactants and swaps the product.
    for d in decoys:
        left, _, right = d.partition(">>")
        assert left == "CCO"
        assert right != "CC=O"


def test_make_scrambled_reactions_excludes_true_product():
    decoys = make_scrambled_reactions(
        "CCO", "CC=O", ["CC=O", "CC(=O)O"], n_decoys=10, seed=0)
    # The true product must never appear as a decoy product.
    assert all(d.split(">>")[1] != "CC=O" for d in decoys)
    assert len(decoys) == 1


def test_make_scrambled_reactions_respects_n_decoys():
    pool = [f"C{'C' * i}O" for i in range(1, 20)]
    decoys = make_scrambled_reactions("CCO", "CC=O", pool, n_decoys=5, seed=1)
    assert len(decoys) == 5


def test_make_scrambled_reactions_empty_pool():
    assert make_scrambled_reactions("CCO", "CC=O", [], n_decoys=5) == []
