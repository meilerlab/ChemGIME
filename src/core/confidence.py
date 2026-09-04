"""
ChemGIME Confidence Calibration (F6)
======================================
Combine Tanimoto chemical similarity with BLAST genomic evidence
to produce a calibrated probability of a true gene-reaction match.

Model
-----
The calibration uses a logistic function:

.. math::

    P(\\text{true match}) = \\sigma(\\beta_0 + \\beta_1 \\cdot S_{tan}
    + \\beta_2 \\cdot \\log_{10}(E) + \\beta_3 \\cdot P_{ident})

where :math:`S_{tan} = 1 - D_{tan}` is the Tanimoto similarity,
:math:`E` is the BLAST e-value, :math:`P_{ident}` is percent identity,
and :math:`\\sigma(x) = 1/(1 + e^{-x})`.

Default coefficients are heuristic priors based on the Split-GEM
benchmark.  When labelled data is available, ``fit_calibration_model``
learns optimal coefficients via logistic regression.

Usage::

    from src.core.confidence import calibrate_confidence

    df_enriched = calibrate_confidence(df_rank, blast_features)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default heuristic coefficients (prior to fitting on labelled data).
# These are reasonable defaults based on the domain:
#   - Higher Tanimoto similarity → higher confidence
#   - Lower e-value (more negative log10) → higher confidence
#   - Higher percent identity → higher confidence
_DEFAULT_BETA = {
    "intercept": -2.0,
    "tanimoto_similarity": 4.0,
    "log10_evalue": -0.3,
    "pident": 0.03,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def compute_confidence(
    tanimoto_distance: float | np.ndarray,
    evalue: float | np.ndarray = 1.0,
    pident: float | np.ndarray = 0.0,
    *,
    beta: Optional[dict] = None,
) -> float | np.ndarray:
    """Compute calibrated confidence probability.

    Parameters
    ----------
    tanimoto_distance : float or array
        Tanimoto distance(s) in [0, 1].
    evalue : float or array
        BLAST e-value(s). Values <= 0 are clamped to 1e-300.
    pident : float or array
        BLAST percent identity in [0, 100].
    beta : dict, optional
        Calibration coefficients. Uses defaults if not provided.

    Returns
    -------
    float or ndarray
        Calibrated probability in [0, 1].
    """
    b = beta or _DEFAULT_BETA

    sim = 1.0 - np.asarray(tanimoto_distance, dtype=float)
    ev = np.asarray(evalue, dtype=float)
    pid = np.asarray(pident, dtype=float)

    # Safe log10(evalue) — clamp to avoid log(0)
    ev_safe = np.clip(ev, 1e-300, None)
    log_ev = np.log10(ev_safe)

    z = (
        b["intercept"]
        + b["tanimoto_similarity"] * sim
        + b["log10_evalue"] * log_ev
        + b["pident"] * pid
    )

    return _sigmoid(z)


def calibrate_confidence(
    df_rank: pd.DataFrame,
    blast_features: Optional[pd.DataFrame] = None,
    *,
    beta: Optional[dict] = None,
    distance_col: str = "tan_distance",
    merge_on: str = "reaction_id",
) -> pd.DataFrame:
    """Add a ``confidence`` column to the ranked DataFrame.

    Parameters
    ----------
    df_rank : pd.DataFrame
        Ranked reactions (must have ``distance_col``).
    blast_features : pd.DataFrame, optional
        BLAST quality features from ``build_scoring_features()`` (must have
        ``best_pident``, ``best_evalue`` columns).  If None, confidence is
        based on Tanimoto distance alone.
    beta : dict, optional
        Calibration coefficients.
    distance_col : str
        Column name for Tanimoto distance.
    merge_on : str
        Column to join df_rank with blast_features. In df_rank this is
        typically "reaction_id"; in blast_features it's "rxn".

    Returns
    -------
    pd.DataFrame
        Copy of df_rank with added ``confidence`` column.
    """
    df = df_rank.copy()

    if blast_features is not None and not blast_features.empty:
        # Aggregate blast features per reaction (take best across proteins)
        blast_agg = (
            blast_features.groupby("rxn")
            .agg(best_pident=("best_pident", "max"), best_evalue=("best_evalue", "min"))
            .reset_index()
        )
        df = df.merge(
            blast_agg,
            left_on=merge_on,
            right_on="rxn",
            how="left",
            suffixes=("", "_blast"),
        )
        evalue = df["best_evalue"].fillna(1.0).values
        pident = df["best_pident"].fillna(0.0).values
    else:
        evalue = np.ones(len(df))
        pident = np.zeros(len(df))

    df["confidence"] = compute_confidence(
        df[distance_col].values,
        evalue=evalue,
        pident=pident,
        beta=beta,
    )

    # Clean up merge artifacts
    for col in ("rxn", "best_pident", "best_evalue"):
        if col in df.columns and col not in df_rank.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    return df


def fit_calibration_model(
    df_labelled: pd.DataFrame,
    *,
    distance_col: str = "tan_distance",
    evalue_col: str = "best_evalue",
    pident_col: str = "best_pident",
    label_col: str = "is_correct",
) -> dict:
    """Fit calibration coefficients from labelled benchmark data.

    Uses scikit-learn's LogisticRegression. The labelled DataFrame must
    have a boolean ``is_correct`` column indicating whether the retrieved
    reaction was a true match.

    Parameters
    ----------
    df_labelled : pd.DataFrame
        Must contain distance, evalue, pident, and label columns.

    Returns
    -------
    dict
        Fitted beta coefficients with keys matching ``_DEFAULT_BETA``.

    Raises
    ------
    ImportError
        If scikit-learn is not available.
    """
    from sklearn.linear_model import LogisticRegression

    df = df_labelled.dropna(subset=[distance_col, label_col]).copy()
    if len(df) < 10:
        logger.warning("Too few labelled samples for calibration fitting; using defaults.")
        return _DEFAULT_BETA.copy()

    sim = (1.0 - df[distance_col]).values
    ev_safe = np.clip(df[evalue_col].fillna(1.0).values, 1e-300, None)
    log_ev = np.log10(ev_safe)
    pid = df[pident_col].fillna(0.0).values
    y = df[label_col].astype(int).values

    X = np.column_stack([sim, log_ev, pid])

    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X, y)

    beta = {
        "intercept": float(lr.intercept_[0]),
        "tanimoto_similarity": float(lr.coef_[0][0]),
        "log10_evalue": float(lr.coef_[0][1]),
        "pident": float(lr.coef_[0][2]),
    }
    logger.info(f"Fitted calibration coefficients: {beta}")
    return beta
