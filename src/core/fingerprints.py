"""
ChemGIME Fingerprint Engine
============================
Reaction fingerprint calculation: DRFP (binary) and QRFP (quaternary).
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MAX_WORKERS

logger = logging.getLogger(__name__)


class FingerprintType(str, Enum):
    """Supported reaction fingerprint types."""
    DRFP = "drfp"
    QRFP = "qrfp"
    # Substrate-only DRFP: encodes the left side alone, ignoring the product.
    # Used for the MicrobeRX ablation that separates ChemGIME's method-only
    # advantage (similarity retrieval) from the extra product information.
    DRFP_SUB = "drfp_sub"


def _calculate_drfp_fingerprint(smiles_pair: Tuple[str, str]) -> Optional[np.ndarray]:
    """
    Calculates the DRFP fingerprint for a (left_smiles, right_smiles) pair.
    """
    try:
        from drfp.fingerprint import DrfpEncoder
    except ImportError:
        raise ImportError(
            "`drfp` library not found. Please install it: `pip install drfp`"
        )

    left_smiles, right_smiles = smiles_pair
    if not left_smiles or not right_smiles:
        return None

    try:
        reaction_smiles = f"{left_smiles}>>{right_smiles}"
        fp = DrfpEncoder.encode(reaction_smiles)[0]
        return fp
    except Exception:
        return None


def _calculate_drfp_substrate_fingerprint(
    smiles_pair: Tuple[str, str],
) -> Optional[np.ndarray]:
    """
    Substrate-only DRFP: encodes ``f"{left_smiles}>>"`` and ignores the
    product, matching MicrobeRX's substrate-only input. Used only for the
    ablation in the MicrobeRX comparison; the product side is intentionally
    dropped so that the full-reaction vs substrate-only gap isolates the
    product-information contribution to retrieval.
    """
    try:
        from drfp.fingerprint import DrfpEncoder
    except ImportError:
        raise ImportError(
            "`drfp` library not found. Please install it: `pip install drfp`"
        )

    left_smiles, _ = smiles_pair
    if not left_smiles:
        return None

    try:
        fp = DrfpEncoder.encode(f"{left_smiles}>>")[0]
        return fp
    except Exception:
        return None


def _calculate_qrfp_fingerprint(smiles_pair: Tuple[str, str]) -> Optional[np.ndarray]:
    """
    Calculates the Quaternary Reaction Fingerprint for a
    (left_smiles, right_smiles) pair.

    Generates three DRFP variants (full, substrate-only, product-only),
    combines them element-wise, and assigns each position a quaternary label:
        0    — no feature
        2048 — substrate-only feature
        2049 — product-only feature
        2050 — feature on both sides

    Returns a 1-D float32 array of length 2048.
    """
    try:
        from drfp.fingerprint import DrfpEncoder
    except ImportError:
        raise ImportError(
            "`drfp` library not found. Please install it: `pip install drfp`"
        )

    left, right = smiles_pair
    if not left or not right:
        return None

    try:
        fp_full = DrfpEncoder.encode(f"{left}>>{right}")[0]
        fp_sub = DrfpEncoder.encode(f"{left}>>")[0]
        fp_pro = DrfpEncoder.encode(f">>{right}")[0]

        combined = fp_sub + fp_full + fp_pro
        combined[combined == 3] = 2
        normalised = 0.5 * combined

        sub_only = normalised - fp_pro
        pro_only = normalised - fp_sub
        all_have = normalised - fp_full

        label = 2048 * sub_only + 2049 * pro_only + 2050 * all_have
        return label.astype(np.float32)
    except Exception:
        return None


def calculate_fingerprints_parallel(
    df: pd.DataFrame,
    fp_type: FingerprintType = FingerprintType.DRFP,
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    Calculates reaction fingerprints in parallel.
    Uses the "raw_left_smiles" and "raw_right_smiles" columns.

    Parameters
    ----------
    fp_type : FingerprintType
        DRFP (binary, default) or QRFP (quaternary, ~3x slower).

    Returns a 2D numpy array of fingerprints and the valid indices.
    """
    if fp_type == FingerprintType.DRFP:
        fn = _calculate_drfp_fingerprint
    elif fp_type == FingerprintType.DRFP_SUB:
        fn = _calculate_drfp_substrate_fingerprint
    else:
        fn = _calculate_qrfp_fingerprint

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        smiles_pairs = list(zip(df['raw_left_smiles'], df['raw_right_smiles']))
        results = list(executor.map(fn, smiles_pairs))

    valid_fps = [fp for fp in results if fp is not None]
    valid_indices = [i for i, fp in enumerate(results) if fp is not None]

    if not valid_fps:
        return None, None

    return np.array(valid_fps), valid_indices
