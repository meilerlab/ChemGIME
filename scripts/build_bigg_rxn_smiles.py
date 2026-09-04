"""
build_bigg_rxn_smiles.py
========================
Pre-generate ``data/database/bigg_rxn_to_smiles.tsv`` from two files that
already ship with ChemGIME:

  1. ``data/database/bigg_reactions.tbl``
     BiGG reaction table with a human-readable ``reaction_string`` column.
     e.g.: ``atp_c + h2o_c --> adp_c + pi_c + h_c``

  2. ``data/database/bigg_to_smiles.tsv``
     Metabolite-level BiGG-ID → SMILES map built by ``create_mapping.py``.

Output
------
``data/database/bigg_rxn_to_smiles.tsv``
Columns: rxn | left_smiles | right_smiles

This file is the pre-installed reaction-SMILES lookup for Gapseq .tbl
standalone mode (``BIGG_RXN_SMILES_FILE`` in config.py).
The ``load_rxn_smiles_map`` loader reads it transparently.

Run once after cloning or updating the database files:

    python scripts/build_bigg_rxn_smiles.py

Or with custom paths:

    python scripts/build_bigg_rxn_smiles.py \\
        --reactions  data/database/bigg_reactions.tbl \\
        --met-smiles data/database/bigg_to_smiles.tsv \\
        --output     data/database/bigg_rxn_to_smiles.tsv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Default paths (relative to project root) ─────────────────────────────────
_HERE = Path(__file__).resolve().parent          # scripts/
_ROOT = _HERE.parent                              # project root
_DB   = _ROOT / "data" / "database"

DEFAULT_REACTIONS  = _DB / "bigg_reactions.tbl"
DEFAULT_MET_SMILES = _DB / "bigg_to_smiles.tsv"
DEFAULT_OUTPUT     = _DB / "bigg_rxn_to_smiles.tsv"

# Matches an optional stoichiometric coefficient at the front of a term:
#   "(2) met_c"  →  strip "(2) "
#   "2 met_c"    →  strip "2 "
#   "met_c"      →  no change
_COEFF_RE = re.compile(r"^\s*(?:\(\s*[\d.]+\s*\)|[\d.]+)\s*")

# Reaction direction separators — ordered longest-first to avoid partial matches
_SEPARATORS = ["<=>", "<->", "-->", "<--"]


# ── Reaction-string parser ────────────────────────────────────────────────────

def _strip_coeff(term: str) -> str:
    """Remove leading stoichiometric coefficient from a metabolite term."""
    return _COEFF_RE.sub("", term).strip()


def _strip_compartment(met_id: str) -> str:
    """Strip trailing single-letter compartment suffix.

    Examples
    --------
    >>> _strip_compartment("atp_c")
    'atp'
    >>> _strip_compartment("glc__D_c")
    'glc__D'
    >>> _strip_compartment("h2o_p")
    'h2o'
    >>> _strip_compartment("23ccmp_e")
    '23ccmp'
    """
    # Compartment is always a single lowercase letter after the last underscore.
    # Use rsplit to split at the last underscore only.
    if "_" in met_id:
        base, suffix = met_id.rsplit("_", 1)
        if len(suffix) == 1 and suffix.isalpha():
            return base
    return met_id


def _parse_side(side_str: str) -> List[str]:
    """Parse one side of a reaction equation into a list of base metabolite IDs."""
    terms = [t.strip() for t in side_str.split("+") if t.strip()]
    ids = []
    for term in terms:
        met_with_comp = _strip_coeff(term)
        if not met_with_comp:
            continue
        ids.append(_strip_compartment(met_with_comp))
    return ids


def parse_reaction_string(
    rxn_str: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """Parse a BiGG reaction_string into (left_ids, right_ids).

    Returns ``None`` for malformed or empty strings.
    """
    if not rxn_str or not isinstance(rxn_str, str):
        return None

    sep_used = None
    for sep in _SEPARATORS:
        if sep in rxn_str:
            sep_used = sep
            break

    if sep_used is None:
        return None

    parts = rxn_str.split(sep_used, 1)
    if len(parts) != 2:
        return None

    left_raw, right_raw = parts

    if sep_used == "<--":
        # Arrow is reversed — swap sides so left = reactants, right = products
        left_raw, right_raw = right_raw, left_raw

    left_ids  = _parse_side(left_raw)
    right_ids = _parse_side(right_raw)

    # Skip exchange / demand / sink reactions (empty side)
    if not left_ids or not right_ids:
        return None

    return left_ids, right_ids


# ── SMILES builder ────────────────────────────────────────────────────────────

def build_reaction_smiles(
    left_ids: List[str],
    right_ids: List[str],
    met_smiles: Dict[str, str],
) -> Optional[Tuple[str, str]]:
    """Look up SMILES for each metabolite and join into dot-separated strings.

    Returns ``None`` if any side ends up with zero known SMILES
    (incomplete coverage is OK — missing metabolites are skipped, but a side
    must not be entirely empty).
    """
    left_smi  = [met_smiles[m] for m in left_ids  if m in met_smiles]
    right_smi = [met_smiles[m] for m in right_ids if m in met_smiles]

    if not left_smi or not right_smi:
        return None

    return ".".join(left_smi), ".".join(right_smi)


# ── Main build logic ──────────────────────────────────────────────────────────

def build(
    reactions_path: Path,
    met_smiles_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Full pipeline: parse BiGG reactions + metabolite SMILES → reaction TSV."""

    # 1. Load metabolite SMILES map
    logger.info(f"Loading metabolite SMILES from {met_smiles_path} …")
    met_df = pd.read_csv(met_smiles_path, sep="\t", dtype=str)
    # Keep only rows where bigg_id and smiles are present
    met_df = met_df.dropna(subset=["bigg_id", "smiles"])
    met_df = met_df.drop_duplicates(subset=["bigg_id"], keep="first")
    met_smiles: Dict[str, str] = met_df.set_index("bigg_id")["smiles"].to_dict()
    logger.info(f"  {len(met_smiles):,} metabolite SMILES loaded.")

    # 2. Load reaction table
    logger.info(f"Loading BiGG reactions from {reactions_path} …")
    rxn_df = pd.read_csv(reactions_path, sep="\t", dtype=str, comment=None)
    logger.info(f"  {len(rxn_df):,} reactions found.")

    if "bigg_id" not in rxn_df.columns or "reaction_string" not in rxn_df.columns:
        raise ValueError(
            f"Expected columns 'bigg_id' and 'reaction_string' in {reactions_path}. "
            f"Got: {list(rxn_df.columns)}"
        )

    # 3. Parse + look up SMILES
    records = []
    counters = {"parsed": 0, "empty_side": 0, "no_smiles": 0, "ok": 0}

    for _, row in rxn_df.iterrows():
        rxn_id  = str(row["bigg_id"]).strip()
        rxn_str = str(row.get("reaction_string", "")).strip()

        parsed = parse_reaction_string(rxn_str)
        if parsed is None:
            counters["empty_side"] += 1
            continue

        counters["parsed"] += 1
        left_ids, right_ids = parsed

        smiles_pair = build_reaction_smiles(left_ids, right_ids, met_smiles)
        if smiles_pair is None:
            counters["no_smiles"] += 1
            continue

        counters["ok"] += 1
        records.append({
            "rxn":          rxn_id,
            "left_smiles":  smiles_pair[0],
            "right_smiles": smiles_pair[1],
        })

    logger.info(
        f"  Parsed OK:       {counters['ok']:,}\n"
        f"  Empty side skip: {counters['empty_side']:,} (exchange/demand reactions)\n"
        f"  No SMILES skip:  {counters['no_smiles']:,} (metabolites missing from map)"
    )

    if not records:
        raise RuntimeError(
            "No reactions with SMILES could be built. "
            "Check that bigg_reactions.tbl and bigg_to_smiles.tsv are compatible."
        )

    result_df = pd.DataFrame(records)

    # 4. Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, sep="\t", index=False)
    logger.info(
        f"Written {len(result_df):,} reaction-SMILES entries to {output_path}"
    )

    return result_df


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build bigg_rxn_to_smiles.tsv from bigg_reactions.tbl + bigg_to_smiles.tsv"
    )
    parser.add_argument(
        "--reactions",  default=str(DEFAULT_REACTIONS),
        help=f"Path to bigg_reactions.tbl  [default: {DEFAULT_REACTIONS}]",
    )
    parser.add_argument(
        "--met-smiles", default=str(DEFAULT_MET_SMILES),
        help=f"Path to bigg_to_smiles.tsv  [default: {DEFAULT_MET_SMILES}]",
    )
    parser.add_argument(
        "--output",     default=str(DEFAULT_OUTPUT),
        help=f"Output TSV path              [default: {DEFAULT_OUTPUT}]",
    )
    args = parser.parse_args()

    build(
        reactions_path=Path(args.reactions),
        met_smiles_path=Path(args.met_smiles),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
