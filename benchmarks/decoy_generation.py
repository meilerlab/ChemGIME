"""
Decoy Generation for ChemGIME Specificity Evaluation
====================================================

A *decoy* is a reaction that is **chemically similar** to a true query yet
**catalysed by a different enzyme class** — exactly the structural look-alike
a permissive similarity threshold would wrongly retrieve.  Ranking the true
target above such decoys is the evidence that ChemGIME captures genuine
enzymatic reactivity rather than bulk structural similarity.

Two complementary strategies are provided (see ``decoy_plan.md`` and
reviewer.md §4.7):

* **Cross-EC GEM decoys** (:func:`label_candidates`).  For each query, every
  reaction already in the GEM is labelled *active* (the reaction the query
  enzyme actually catalyses) or *decoy* (a different enzyme class).  No
  external data and no synthetic molecules are required, so it runs on all
  130 M-CSA queries.  This is the workhorse and supplies the headline
  ROC-AUC / enrichment numbers.

* **Scrambled-reaction decoys** (:func:`make_scrambled_reactions`).  The true
  reactants are kept but paired with the products of a different, chemically
  related reaction, yielding a plausible-looking yet chemically invalid
  reaction.  This probes whether the reaction fingerprint encodes the actual
  transformation rather than substrate identity alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════
#  Shared EC utilities
# ════════════════════════════════════════════════════════════════════════

def ec_matches(query_ec: str, candidate_ec: str, level: int = 3) -> bool:
    """True if two EC numbers agree on their first ``level`` fields.

    Wildcard fields (``-``) and blanks never match.  ``level`` ranges 1-4
    (1 = enzyme class, 4 = fully specified serial number).
    """
    if not query_ec or not candidate_ec:
        return False
    q = query_ec.strip().split(".")
    c = candidate_ec.strip().split(".")
    if len(q) < level or len(c) < level:
        return False
    for i in range(level):
        qi, ci = q[i].strip(), c[i].strip()
        if not qi.isdigit() or not ci.isdigit() or qi != ci:
            return False
    return True


def _any_ec_matches(query_ec: str, ec_field: str, level: int) -> bool:
    """Match a query EC against a candidate's (possibly multi-EC) field.

    GEM ``ec`` fields hold one or more EC numbers separated by ``;`` or ``,``.
    """
    if not ec_field or ec_field.lower() == "nan":
        return False
    for ec in str(ec_field).replace(",", ";").split(";"):
        if ec_matches(query_ec, ec, level=level):
            return True
    return False


# ════════════════════════════════════════════════════════════════════════
#  Strategy 2 — cross-EC GEM decoys
# ════════════════════════════════════════════════════════════════════════

@dataclass
class LabelledCandidates:
    """Per-candidate decoy labelling aligned to a ranked GEM DataFrame.

    All arrays share the row order of the ranked DataFrame (sorted by
    ascending Tanimoto distance).  ``similarity = 1 - tan_distance``.

    Categories
    ----------
    active
        A true target reaction for this query (precise ground truth: the
        reaction(s) the query enzyme actually catalyses).
    decoy_hard
        A *different* enzyme class AND chemically similar
        (``similarity >= sim_threshold``) — i.e. a structural look-alike that
        falls inside ChemGIME's operational voting window.  These are the
        false positives a permissive threshold would wrongly accept.
    decoy_far
        A different enzyme class but chemically dissimilar (below threshold).
        Part of the global decoy set, not the hard within-window set.
    ambiguous
        The *same* EC class but not a labelled active (e.g. an isozyme /
        alternative annotation).  Excluded from both decoy sets because
        retrieving it is neither clearly right nor clearly wrong.

    Two decoy sets are exposed:
      * :meth:`hard_subset` — active vs ``decoy_hard`` (stringent, tests
        discrimination among close look-alikes).
      * :meth:`global_subset` — active vs all different-class reactions
        (``decoy_hard`` + ``decoy_far``); the broad enzyme-retrieval test.
    """
    category: np.ndarray        # str
    is_active: np.ndarray       # bool, full list
    is_decoy_hard: np.ndarray   # bool, within-window different-class
    is_decoy_all: np.ndarray    # bool, any-distance different-class
    similarity: np.ndarray      # float, full list (higher = closer)
    reaction_id: np.ndarray     # str, full list
    n_active: int = 0
    n_decoy: int = 0            # hard decoys (within window)
    n_decoy_all: int = 0
    n_ambiguous: int = 0

    def hard_subset(self) -> tuple[np.ndarray, np.ndarray]:
        """``(labels, scores)`` for active vs within-window decoys."""
        mask = self.is_active | self.is_decoy_hard
        return self.is_active[mask].astype(int), self.similarity[mask]

    def global_subset(self) -> tuple[np.ndarray, np.ndarray]:
        """``(labels, scores)`` for active vs all different-class reactions."""
        mask = self.is_active | self.is_decoy_all
        return self.is_active[mask].astype(int), self.similarity[mask]


def label_candidates(
    ranked: pd.DataFrame,
    active_rxn_ids: Iterable[str],
    *,
    target_ec: Optional[str] = None,
    decoy_level: int = 3,
    sim_threshold: float = 0.5,
    exclude_rxn_ids: Optional[Iterable[str]] = None,
) -> LabelledCandidates:
    """Label every GEM reaction as active / decoy / ambiguous / easy-negative.

    Parameters
    ----------
    ranked
        Output of :func:`src.core.similarity.get_closest_rxns` — must carry
        ``tan_distance``, ``ec`` and ``reaction_id`` columns.
    active_rxn_ids
        Reaction IDs that are the *true target(s)* for this query, determined
        upstream by GPR/UniProt match or exact (L4) EC match.  This precise
        active set is what makes the discrimination test meaningful: a broad
        same-subclass set would mislabel chemically-distant reactions as
        actives and spuriously depress the AUC.
    target_ec
        Query EC.  Used only to spare same-class reactions from the decoy
        label (an isozyme of the true enzyme is not a false positive).
    decoy_level
        EC level at which a similar reaction must *differ* from the query to
        count as a decoy (default 3 = sub-subclass).
    sim_threshold
        Minimum similarity (``1 - tan_distance``) for a non-target reaction to
        qualify as a *hard* decoy.  Below it -> easy negative.
    exclude_rxn_ids
        Reaction IDs to drop entirely (e.g. an exact self-match for a LOO run).
    """
    if "tan_distance" not in ranked.columns:
        raise ValueError("ranked DataFrame must contain a 'tan_distance' column")

    actives = set(active_rxn_ids)
    exclude = set(exclude_rxn_ids or ())
    n = len(ranked)

    category = np.empty(n, dtype=object)
    is_active = np.zeros(n, dtype=bool)
    is_decoy_hard = np.zeros(n, dtype=bool)
    is_decoy_all = np.zeros(n, dtype=bool)
    similarity = (1.0 - ranked["tan_distance"].to_numpy(dtype=float))
    rxn_ids = ranked.get("reaction_id", pd.Series([""] * n)).astype(str).to_numpy()
    ec_field = ranked.get("ec", pd.Series([""] * n)).astype(str).to_numpy()

    for i in range(n):
        rid = rxn_ids[i]
        if rid in exclude:
            category[i] = "excluded"
            similarity[i] = np.nan  # keep alignment but never selected
            continue

        if rid in actives:
            category[i] = "active"
            is_active[i] = True
            continue

        # Same EC class but not a target -> ambiguous (isozyme-like), excluded.
        if target_ec and _any_ec_matches(target_ec, ec_field[i], decoy_level):
            category[i] = "ambiguous"
            continue

        # Different enzyme class -> a decoy. Hard if inside the voting window.
        is_decoy_all[i] = True
        if similarity[i] >= sim_threshold:
            category[i] = "decoy_hard"
            is_decoy_hard[i] = True
        else:
            category[i] = "decoy_far"

    return LabelledCandidates(
        category=category,
        is_active=is_active,
        is_decoy_hard=is_decoy_hard,
        is_decoy_all=is_decoy_all,
        similarity=similarity,
        reaction_id=rxn_ids,
        n_active=int(is_active.sum()),
        n_decoy=int(is_decoy_hard.sum()),
        n_decoy_all=int(is_decoy_all.sum()),
        n_ambiguous=int(np.sum(category == "ambiguous")),
    )


# ════════════════════════════════════════════════════════════════════════
#  Strategy 1 — scrambled-reaction decoys (reviewer.md §4.7 i)
# ════════════════════════════════════════════════════════════════════════
#
# A reaction fingerprint encodes the *transformation* (substrate -> product),
# not just the substrate.  A scrambled decoy keeps the true reactants but
# swaps in the products of a different, chemically related reaction.  The
# decoy is therefore a plausible-looking but chemically invalid reaction.  If
# ChemGIME's fingerprint captures genuine reactivity, the true reaction should
# find a closer match in the GEM than its scrambled counterparts.


def make_scrambled_reactions(
    true_left: str,
    true_right: str,
    donor_products: List[str],
    *,
    n_decoys: int = 10,
    seed: int = 42,
) -> List[str]:
    """Build scrambled decoy reactions ``true_left >> donor_product``.

    ``donor_products`` is a pool of product-side SMILES drawn from other
    (chemically related) reactions.  Donors identical to ``true_right`` are
    skipped so a decoy is never the true reaction.  Returns up to ``n_decoys``
    distinct ``"left>>right"`` reaction SMILES.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    pool = [p for p in dict.fromkeys(donor_products)  # de-dup, keep order
            if p and p != true_right]
    if not pool:
        return []
    rng.shuffle(pool)
    return [f"{true_left}>>{p}" for p in pool[:n_decoys]]
