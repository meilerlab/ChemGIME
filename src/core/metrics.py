"""
ChemGIME Metrics
================
Retrieval-quality metrics for benchmarking and reporting.

Key metric: **Mean Reciprocal Rank (MRR)**

.. math::

    MRR = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{rank_i}

where :math:`rank_i` is the position at which the correct answer first
appears for query *i*.  Queries where the answer was not found contribute
0 to the sum (reciprocal rank = 0).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def reciprocal_rank(rank: Optional[int]) -> float:
    """Return 1/rank, or 0 if the answer was not found (rank is None)."""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / rank


def mean_reciprocal_rank(
    ranks: Sequence[Optional[int]],
    *,
    ignore_missing: bool = False,
) -> float:
    """Compute Mean Reciprocal Rank over a set of queries.

    Parameters
    ----------
    ranks : sequence of int or None
        The rank at which the correct answer was found for each query.
        ``None`` means the answer was not found (contributes 0.0).
    ignore_missing : bool
        If True, queries with rank=None are excluded from the denominator
        (only scored queries count).  If False (default), missing answers
        reduce the MRR by increasing |Q|.

    Returns
    -------
    float
        MRR in [0, 1].  1.0 = every correct answer was ranked #1.
    """
    if not ranks:
        return 0.0

    rrs = [reciprocal_rank(r) for r in ranks]

    if ignore_missing:
        scored = [rr for rr in rrs if rr > 0]
        return float(np.mean(scored)) if scored else 0.0

    return float(np.mean(rrs))


def top_k_accuracy(
    ranks: Sequence[Optional[int]],
    k: int,
) -> float:
    """Compute Top-k accuracy (percentage of queries found within rank k).

    Parameters
    ----------
    ranks : sequence of int or None
        Rank at which correct answer was found per query.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Accuracy as a percentage in [0, 100].
    """
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks) * 100


def retrieval_summary(
    ranks: Sequence[Optional[int]],
    k_values: tuple = (1, 5, 10),
) -> dict:
    """Compute a full retrieval summary: MRR + Top-k for multiple k.

    Returns
    -------
    dict
        Keys: ``mrr``, ``top1``, ``top5``, ``top10``, ``n_queries``,
        ``n_found``, ``n_missing``.
    """
    n = len(ranks)
    n_found = sum(1 for r in ranks if r is not None and r > 0)

    summary = {
        "n_queries": n,
        "n_found": n_found,
        "n_missing": n - n_found,
        "mrr": mean_reciprocal_rank(ranks),
    }
    for k in k_values:
        summary[f"top{k}"] = top_k_accuracy(ranks, k)

    return summary
