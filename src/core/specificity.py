"""
ChemGIME Specificity / Decoy-Discrimination Metrics
===================================================

The retrieval metrics in :mod:`src.core.metrics` (MRR, Top-k) measure
**sensitivity**: given the true enzyme, how highly is it ranked?  They are
silent on **specificity**: when chemically similar but enzymatically wrong
reactions ("decoys") are present, does ChemGIME still rank the true target
above them?

This module supplies the standard virtual-screening discrimination metrics
used to answer that question:

  * **ROC-AUC** — probability that a true active is scored above a decoy.
  * **Enrichment Factor (EF)** — fold-enrichment of actives in the top x%.
  * **BEDROC** — Boltzmann-enhanced early-recognition metric (Truchon &
    Bayly, *J. Chem. Inf. Model.* 2007).
  * **Precision@k** — fraction of the top-k retrieved that are true actives.
  * **FPR/TPR at an operational threshold** — false-/true-positive rate at
    the similarity cut-off ChemGIME actually uses for EC voting.

Sign convention
---------------
Every function takes a **score where higher means "more likely active"**.
ChemGIME ranks by Tanimoto/Jaccard *distance* (lower is better), so callers
pass ``score = 1 - tan_distance`` (a similarity). This keeps the metric layer
independent of ChemGIME's internal distance representation.

Implementations wrap scikit-learn (ROC-AUC, ROC curve, average precision) and
RDKit's validated ``rdkit.ML.Scoring`` module (BEDROC, enrichment) so the
numbers are reproducible against tools reviewers already trust.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

__all__ = [
    "roc_auc",
    "roc_points",
    "average_precision",
    "enrichment_factor",
    "bedroc",
    "precision_at_k",
    "recall_at_k",
    "rates_at_threshold",
    "discrimination_summary",
    "bootstrap_ci",
]


def _as_arrays(labels: Sequence[int], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D arrays and validate shape/content."""
    y = np.asarray(labels, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"labels and scores length mismatch: {y.shape} vs {s.shape}")
    if y.size == 0:
        raise ValueError("empty labels/scores")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be binary (0/1)")
    return y, s


def _both_classes_present(y: np.ndarray) -> bool:
    """ROC/AP are undefined unless both an active and a decoy are present."""
    return 0 < int(y.sum()) < y.size


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Area under the ROC curve (active=1 ranked above decoy=0).

    Returns ``None`` when only one class is present (AUC undefined for a
    query that has no decoys or no actives).
    """
    y, s = _as_arrays(labels, scores)
    if not _both_classes_present(y):
        return None
    return float(roc_auc_score(y, s))


def roc_points(
    labels: Sequence[int], scores: Sequence[float]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(fpr, tpr, thresholds)`` for a ROC curve, or ``None``.

    Thin wrapper over :func:`sklearn.metrics.roc_curve`.
    """
    y, s = _as_arrays(labels, scores)
    if not _both_classes_present(y):
        return None
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Average precision (area under the precision-recall curve)."""
    y, s = _as_arrays(labels, scores)
    if not _both_classes_present(y):
        return None
    return float(average_precision_score(y, s))


def enrichment_factor(
    labels: Sequence[int], scores: Sequence[float], fraction: float = 0.01
) -> Optional[float]:
    """Enrichment factor at the top ``fraction`` of the ranked list.

    .. math::

        EF_{x\\%} = \\frac{a_x / n_x}{A / N}

    where :math:`n_x = \\lceil x\\,N \\rceil` is the number of candidates in
    the top fraction, :math:`a_x` the actives among them, :math:`A` the total
    actives, and :math:`N` the total candidates.  ``EF = 1`` is random; the
    maximum attainable value is :math:`1/x` (when every top-ranked item is an
    active).  Returns ``None`` if there are no actives.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y, s = _as_arrays(labels, scores)
    n = y.size
    n_actives = int(y.sum())
    if n_actives == 0:
        return None

    n_top = max(1, int(np.ceil(fraction * n)))
    order = np.argsort(-s, kind="mergesort")  # stable, descending score
    actives_in_top = int(y[order[:n_top]].sum())

    return float((actives_in_top / n_top) / (n_actives / n))


def bedroc(
    labels: Sequence[int], scores: Sequence[float], alpha: float = 20.0
) -> Optional[float]:
    """BEDROC score (early-recognition, Truchon & Bayly 2007).

    ``alpha=20`` (default) weights the top ~8% of the ranking most heavily,
    the convention in ligand-based virtual screening.  Computed via RDKit's
    validated ``rdkit.ML.Scoring.Scoring.CalcBEDROC``.  Returns ``None`` when
    only one class is present.
    """
    y, s = _as_arrays(labels, scores)
    if not _both_classes_present(y):
        return None

    from rdkit.ML.Scoring import Scoring

    order = np.argsort(-s, kind="mergesort")  # descending score
    ranked = [[int(lbl)] for lbl in y[order]]
    return float(Scoring.CalcBEDROC(ranked, 0, alpha))


def precision_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Fraction of the top-``k`` highest-scoring candidates that are actives.

    If ``k`` exceeds the number of candidates it is clamped to that number.
    Directly answers "for the candidates ChemGIME surfaces, how many are real
    hits versus structural look-alikes?"
    """
    if k <= 0:
        raise ValueError("k must be positive")
    y, s = _as_arrays(labels, scores)
    k_eff = min(k, y.size)
    order = np.argsort(-s, kind="mergesort")
    return float(y[order[:k_eff]].sum() / k_eff)


def recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> Optional[float]:
    """Fraction of all actives recovered within the top-``k``."""
    if k <= 0:
        raise ValueError("k must be positive")
    y, s = _as_arrays(labels, scores)
    n_actives = int(y.sum())
    if n_actives == 0:
        return None
    k_eff = min(k, y.size)
    order = np.argsort(-s, kind="mergesort")
    return float(y[order[:k_eff]].sum() / n_actives)


def rates_at_threshold(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict:
    """True-/false-positive rates when accepting every candidate scoring
    ``>= threshold``.

    ``threshold`` is on the *score* (similarity).  For ChemGIME's operational
    distance cut-off ``tau`` pass ``threshold = 1 - tau`` (e.g. tau=0.5 ->
    threshold=0.5).  Returns TPR, FPR, precision and the raw confusion counts.
    """
    y, s = _as_arrays(labels, scores)
    accepted = s >= threshold
    tp = int(np.sum(accepted & (y == 1)))
    fp = int(np.sum(accepted & (y == 0)))
    fn = int(np.sum(~accepted & (y == 1)))
    tn = int(np.sum(~accepted & (y == 0)))

    tpr = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    precision = tp / (tp + fp) if (tp + fp) else None

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tpr, "fpr": fpr, "precision": precision,
    }


def discrimination_summary(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    k_values: Sequence[int] = (1, 5, 10),
    ef_fractions: Sequence[float] = (0.01, 0.05),
    bedroc_alpha: float = 20.0,
    sim_threshold: float = 0.5,
) -> dict:
    """Full single-query discrimination report.

    Bundles ROC-AUC, average precision, BEDROC, enrichment factors,
    precision@k and the rates at the operational threshold into one dict.
    Metrics that are undefined for the input (single-class) are ``None``.
    """
    y, s = _as_arrays(labels, scores)
    out = {
        "n_candidates": int(y.size),
        "n_actives": int(y.sum()),
        "n_decoys": int(y.size - y.sum()),
        "roc_auc": roc_auc(y, s),
        "average_precision": average_precision(y, s),
        "bedroc": bedroc(y, s, alpha=bedroc_alpha),
    }
    for f in ef_fractions:
        out[f"ef_{f:g}"] = enrichment_factor(y, s, fraction=f)
    for k in k_values:
        out[f"precision_at_{k}"] = precision_at_k(y, s, k)
    out["at_threshold"] = rates_at_threshold(y, s, sim_threshold)
    return out


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(mean, ci_lower, ci_upper)``.  NaNs are dropped first (queries
    with an undefined metric do not contribute).
    """
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.RandomState(seed)
    boot = np.array([
        rng.choice(arr, size=arr.size, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(arr.mean()), float(lo), float(hi)
