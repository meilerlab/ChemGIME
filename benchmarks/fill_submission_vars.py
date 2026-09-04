"""
fill_submission_vars.py — Read benchmark JSON results and update submission_vars.tex

Usage:
    python benchmarks/fill_submission_vars.py
    python benchmarks/fill_submission_vars.py --dry-run   # print without writing
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VARS_FILE = PROJECT_ROOT / "manuscript" / "submission_vars.tex"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [SKIP] not found: {path}")
        return None
    return json.load(open(path))


def _patch(content: str, command: str, value: str) -> str:
    """Replace \\newcommand{\\CMD}{...} value inline."""
    pattern = rf'(\\newcommand\{{\\{command}\}}\{{)[^}}]*(}})'
    replacement = rf'\g<1>{value}\g<2>'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        print(f"  [WARN] command \\{command} not found in submission_vars.tex")
    else:
        print(f"  [OK]   \\{command} -> {value}")
    return new_content


def fill_microberx(content: str) -> str:
    stats_path = DATA_DIR / "microberx_mcsa_statistics_all_cutoff0.6.json"
    s = _load_json(stats_path)
    if s is None:
        return content

    cov = f"{s['coverage_pct']:.1f}\\,\\% ({s['n_evaluable']}/130)"
    mrr = f"{s['mrr']:.3f}"
    top1 = f"{s['top1_pct']:.1f}"
    top5 = f"{s['top5_pct']:.1f}"
    top10 = f"{s['top10_pct']:.1f}"

    content = _patch(content, "MRXcov", cov)
    content = _patch(content, "MRXmrr", mrr)
    content = _patch(content, "MRXtopOne", top1)
    content = _patch(content, "MRXtopFive", top5)
    content = _patch(content, "MRXtopTen", top10)
    return content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    content = VARS_FILE.read_text(encoding="utf-8")
    original = content

    print("\n=== Filling MicrobeRX head-to-head benchmark values ===")
    content = fill_microberx(content)

    if content == original:
        print("\nNo changes — all values already filled or no result files found.")
        return

    if args.dry_run:
        print("\n--- dry run: resulting submission_vars.tex ---")
        for line in content.splitlines():
            if "\\MRX" in line or "\\Scaffold" in line:
                print(" ", line)
    else:
        VARS_FILE.write_text(content, encoding="utf-8")
        print(f"\nWritten: {VARS_FILE}")


if __name__ == "__main__":
    main()
