"""
Characterise CLIPZyme's 58 M-CSA encoding failures by EC class, and report
ChemGIME's retrieval performance on them.

Addresses a reviewer concern on the CLIPZyme comparison (Sec. "Complementarity
with CLIPZyme"): the matched-subset metrics are computed only on the 72 queries
CLIPZyme could encode. If CLIPZyme's 58 failures were systematically easier for
ChemGIME, the matched MRR gap would be inflated. This script tests that.

Two independent questions:
  1. Are CLIPZyme's encoding failures EC-class-biased relative to its successes?
     -> chi-square statistic on the EC-L1 x {encoded, failed} table, with a
        Monte-Carlo (label-permutation) p-value because EC 5/6/7 have zero cells
        and the asymptotic chi-square is unreliable there. Cramer's V as effect
        size.
  2. Does ChemGIME perform differently on the 58 failures vs the 72 matched?
     -> Mann-Whitney U on the per-query reciprocal ranks of the two groups.

DATA SOURCING NOTE
------------------
clipzyme_mcsa_results.json stores chemgime_rr = 0.0 / chemgime_rank = null for
the 58 failures: those are PLACEHOLDERS (the file is CLIPZyme-centric and never
populated ChemGIME ranks outside the matched set). ChemGIME's true per-query
scores are read from the novelty source file instead. Using the placeholders
would wrongly score ChemGIME 0.0 on the 58.
"""

import collections
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

DATA = Path(__file__).resolve().parent / "data"
RESULTS = DATA / "clipzyme_mcsa_results.json"
# Source of truth for ChemGIME per-query ranks (see DATA SOURCING NOTE):
NOVELTY = (
    Path(__file__).resolve().parent.parent
    / "tests" / "benchmark_output" / "mcsa_novelty_iML1515_qrfp_kept.json"
)
OUT = DATA / "clipzyme_failure_characterization.json"

SEED = 42
N_PERM = 100_000

EC_L1_NAMES = {
    "1": "oxidoreductases", "2": "transferases", "3": "hydrolases",
    "4": "lyases", "5": "isomerases", "6": "ligases", "7": "translocases",
}


def ec_l1(ec):
    return ec.split(".")[0] if ec else "?"


def chi2_stat(table):
    """Pearson chi-square statistic for an r x c table (no p-value)."""
    table = np.asarray(table, dtype=float)
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    total = table.sum()
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def monte_carlo_chi2(labels, ec_classes, n_perm, seed):
    """
    Monte-Carlo p-value for independence of (encoded/failed) and EC-L1.

    Keeps both margins of the contingency table fixed in expectation by
    permuting the encoded/failed labels across queries and recomputing the
    chi-square statistic. Robust to the zero cells that make the asymptotic
    chi-square invalid here.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(set(ec_classes))
    cls_idx = {c: i for i, c in enumerate(classes)}
    ec_col = np.array([cls_idx[c] for c in ec_classes])
    labels = np.asarray(labels, dtype=int)  # 1 = failed, 0 = encoded

    def build(lab):
        t = np.zeros((len(classes), 2), dtype=float)
        for ci, li in zip(ec_col, lab):
            t[ci, li] += 1
        return t

    observed = chi2_stat(build(labels))
    count = 0
    for _ in range(n_perm):
        if chi2_stat(build(rng.permutation(labels))) >= observed - 1e-9:
            count += 1
    p = (count + 1) / (n_perm + 1)  # add-one (never reports p = 0)

    # Cramer's V on the observed table
    table = build(labels)
    n = table.sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(observed / (n * k))) if k > 0 else float("nan")
    return observed, p, cramers_v, classes


def metrics(detail_by_id, ids):
    rr = [detail_by_id[i]["reciprocal_rank"] for i in ids]
    found = [detail_by_id[i]["found"] for i in ids]
    ranks = [detail_by_id[i]["rank"] for i in ids]
    n = len(ids)
    top = lambda k: 100.0 * sum(1 for r in ranks if r and r <= k) / n
    return {
        "n": n,
        "mrr": round(sum(rr) / n, 4),
        "found_pct": round(100.0 * sum(found) / n, 1),
        "top1_pct": round(top(1), 1),
        "top5_pct": round(top(5), 1),
        "top10_pct": round(top(10), 1),
        "rr": rr,
    }


def main():
    per_query = json.loads(RESULTS.read_text())["per_query"]
    detail_by_id = {
        d["mcsa_id"]: d
        for d in json.loads(NOVELTY.read_text())["details"]
    }

    encoded = [q for q in per_query if q["encoding_ok"]]
    failed = [q for q in per_query if not q["encoding_ok"]]
    enc_ids = [q["mcsa_id"] for q in encoded]
    fail_ids = [q["mcsa_id"] for q in failed]

    assert all(i in detail_by_id for i in enc_ids + fail_ids), \
        "Some M-CSA ids missing from the novelty source file"

    # --- EC-L1 contingency: rows = EC class, cols = (encoded, failed) ---
    ec_by = {q["mcsa_id"]: ec_l1(q["ec"]) for q in per_query}
    enc_counts = collections.Counter(ec_by[i] for i in enc_ids)
    fail_counts = collections.Counter(ec_by[i] for i in fail_ids)
    all_classes = sorted(set(enc_counts) | set(fail_counts))

    contingency = {
        EC_L1_NAMES.get(c, c): {
            "ec_l1": c,
            "encoded": enc_counts.get(c, 0),
            "failed": fail_counts.get(c, 0),
            "total": enc_counts.get(c, 0) + fail_counts.get(c, 0),
            "failure_rate_pct": round(
                100.0 * fail_counts.get(c, 0)
                / max(1, enc_counts.get(c, 0) + fail_counts.get(c, 0)), 1),
        }
        for c in all_classes
    }

    labels = [1] * len(fail_ids) + [0] * len(enc_ids)
    ec_classes = [ec_by[i] for i in fail_ids] + [ec_by[i] for i in enc_ids]
    chi2, p_mc, cramers_v, _ = monte_carlo_chi2(labels, ec_classes, N_PERM, SEED)

    # --- ChemGIME performance on each group ---
    m_failed = metrics(detail_by_id, fail_ids)
    m_encoded = metrics(detail_by_id, enc_ids)

    # Does ChemGIME perform differently on the two groups?
    u, p_u = mannwhitneyu(
        m_failed["rr"], m_encoded["rr"], alternative="two-sided"
    )

    out = {
        "description": "CLIPZyme encoding-failure characterisation; "
                       "ChemGIME ranks sourced from " + NOVELTY.name,
        "n_encoded": len(enc_ids),
        "n_failed": len(fail_ids),
        "ec_contingency": contingency,
        "ec_independence_test": {
            "test": "Pearson chi-square, Monte-Carlo label permutation",
            "chi2": round(chi2, 3),
            "p_value": round(p_mc, 5),
            "cramers_v": round(cramers_v, 3),
            "n_permutations": N_PERM,
            "seed": SEED,
            "note": "Monte-Carlo because EC 5/6/7 have zero cells",
        },
        "chemgime_on_failed": {k: v for k, v in m_failed.items() if k != "rr"},
        "chemgime_on_encoded": {k: v for k, v in m_encoded.items() if k != "rr"},
        "chemgime_group_difference_test": {
            "test": "Mann-Whitney U (two-sided) on per-query reciprocal ranks",
            "u_statistic": round(float(u), 1),
            "p_value": round(float(p_u), 4),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))

    # --- console summary ---
    print(f"Encoded (matched): {len(enc_ids)}   Failed: {len(fail_ids)}\n")
    print("EC-L1   name              encoded  failed  fail-rate%")
    for name, row in contingency.items():
        print(f"  {row['ec_l1']}    {name:16s}  {row['encoded']:6d}  "
              f"{row['failed']:6d}   {row['failure_rate_pct']:5.1f}")
    print(f"\nchi-square = {chi2:.2f}, Monte-Carlo p = {p_mc:.5f}, "
          f"Cramer's V = {cramers_v:.3f}")
    print(f"\nChemGIME on 58 failed : MRR {m_failed['mrr']}, "
          f"found {m_failed['found_pct']}%, Top-1 {m_failed['top1_pct']}%, "
          f"Top-10 {m_failed['top10_pct']}%")
    print(f"ChemGIME on 72 matched: MRR {m_encoded['mrr']}, "
          f"found {m_encoded['found_pct']}%, Top-1 {m_encoded['top1_pct']}%, "
          f"Top-10 {m_encoded['top10_pct']}%")
    print(f"\nMann-Whitney U = {u:.1f}, p = {p_u:.4f} "
          f"(two-sided; H0: same ChemGIME RR distribution)")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
