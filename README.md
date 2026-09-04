# ChemGIME

**Chemical Genomic Insight into Metabolic Enzymes** — organism-specific enzyme
identification through reaction fingerprinting of genome-scale metabolic models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](environment.yml)

ChemGIME identifies **which enzyme within a specific organism** catalyses a
given chemical transformation. Given a substrate–product reaction (as SMILES)
and an organism-specific genome-scale metabolic model (GEM) in SBML format, it
encodes the full reaction as a **Differential Reaction Fingerprint (DRFP)** or
**Quaternary Reaction Fingerprint (QRFP)**, ranks the GEM's reactions by
Tanimoto similarity, predicts EC numbers by similarity-weighted voting, and
optionally fuses BLASTp sequence evidence into a single confidence score.

The key idea is to use the GEM itself as the search space: every candidate is a
curated, mass-balanced reaction from one organism's network, so retrieval is
confined to that organism's chemistry — no GPU, training, or atom-mapped
reactions required.

> This repository accompanies the manuscript *"ChemGIME: organism-specific
> enzyme identification through reaction fingerprinting of genome-scale
> metabolic models"* (Elaldi et al.). It contains the source code, the two
> reference GEMs, the essential reference data, and every benchmark script and
> result file needed to reproduce the reported numbers.

---

## Installation

ChemGIME uses a pinned conda environment (Python 3.13, RDKit, drfp, COBRApy,
scikit-learn, NCBI BLAST+, Streamlit).

```bash
git clone https://github.com/meilerlab/ChemGIME.git
cd ChemGIME

conda env create -f environment.yml     # or: make env   (Linux / macOS)
conda activate chemgime
```

**On Windows**, use the Windows environment file instead (see
[Platform support](#platform-support)):

```bash
conda env create -f environment-windows.yml
conda activate chemgime
```

## Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | Fully supported (reference platform) | All benchmarks were run here; `environment.yml` and `make all` work as-is. |
| **macOS** | Supported | Same `environment.yml`; BLAST is available from bioconda. |
| **Windows (WSL2)** | Recommended for Windows | Use `environment.yml` inside WSL2 — identical to the Linux path, including BLAST and `make`. |

The code itself is cross-platform (it uses `pathlib` throughout, Windows-safe
temp files, and `shutil.which` guards around external binaries). Two things
differ on **native Windows**:

1. **BLAST is not installed by conda.** The `blast` package is published on
   bioconda for Linux/macOS only, so `environment-windows.yml` omits it.
   BLAST is **optional** — reaction-fingerprint retrieval, EC prediction, and
   the Streamlit interface all run without it. To enable the optional BLASTp
   homology evidence, install
   [NCBI BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/)
   from the official Windows installer and put `blastp` / `makeblastdb` on your
   `PATH`.
2. **`make` is not available by default.** The `Makefile` reproduction targets
   need GNU Make and a POSIX shell. On native Windows, either run the
   underlying commands directly (e.g.
   `python tests/benchmarking_suite.py split-gem --gem iML1515.xml --fingerprint qrfp --filter_cofactors false`),
   or use WSL2 / Git Bash with `make`.

> If in doubt on Windows, use **WSL2** — it gives the exact Linux environment
> with no workarounds.

## Quick start — web interface

```bash
streamlit run src/app/main.py
```

Then upload an SBML GEM (or use the bundled `iML1515.xml` / `iJN746.xml`),
enter a reaction as SMILES, and inspect the ranked candidate enzymes with their
confidence scores.

## Command-line benchmarking

All retrieval benchmarks are driven by a single entry point:

```bash
python tests/benchmarking_suite.py <command> --gem iML1515.xml \
    --fingerprint {drfp|qrfp|drfp_sub} --filter_cofactors {true|false} --seed 42
```

| Command          | Description                                              |
|------------------|----------------------------------------------------------|
| `split-gem`      | 80/20 Split-GEM GPR retrieval benchmark                  |
| `mcsa-novelty`   | M-CSA cross-database EC retrieval                        |
| `mcsa-loo`       | M-CSA leave-one-enzyme-out (strict generalisation)       |
| `cross-validate` | EC-stratified *k*-fold cross-validation (`--folds 5`)    |
| `all`            | Run split-gem, mcsa-novelty and cross-validate in order  |

`--filter_cofactors false` keeps ubiquitous cofactors (the recommended,
best-performing setting); `true` removes them.

## Reproducing the paper

The `Makefile` regenerates every benchmark table and figure from the raw
inputs with a fixed random seed:

```bash
make all        # mcsa + loo + split + cv + simmer + decoy + scramble
make split      # Split-GEM retrieval (both GEMs, all FP/cofactor conditions)
make mcsa       # M-CSA cross-database retrieval
make cv         # stratified 5-fold cross-validation
make decoy      # decoy discrimination benchmark
make scramble   # scrambled-reaction control
make simmer     # SIMMER drug-metabolism benchmark
make test       # unit tests (pytest)
```

Result files are written to `tests/benchmark_output/` and `benchmarks/data/`;
the versions committed here are the ones reported in the manuscript.

## Repository layout

```
ChemGIME/
├── src/
│   ├── core/            # fingerprints, similarity, EC prediction, confidence, GEM parsing …
│   ├── app/             # Streamlit web interface (main.py)
│   └── utils/           # helper scripts
├── benchmarks/          # comparison + evaluation scripts and their result files
│   └── data/            # committed benchmark result JSON/CSV/figures
├── tests/               # unit tests + benchmarking_suite.py + benchmark_output/
├── scripts/             # data-preparation utilities (BiGG→SMILES mapping, etc.)
├── data/database/       # reference data: BiGG↔SMILES maps, cofactors, M-CSA cache, ESM-2 embeddings
├── iML1515.xml          # E. coli K-12 iML1515 GEM (BiGG)
├── iJN746.xml           # P. putida KT2440 iJN746 GEM (BiGG)
├── environment.yml      # pinned conda environment
├── Makefile             # one-command reproducibility
├── CITATION.cff
└── LICENSE
```

## Data notes

- The two reference GEMs (`iML1515.xml`, `iJN746.xml`) are redistributed here
  for convenience and are also available from [BiGG Models](http://bigg.ucsd.edu/).
- `data/database/bigg_to_smiles.tsv` is a pre-computed BiGG-metabolite → SMILES
  map. It is **derived** from the large MetaNetX tables `chem_prop.tsv` and
  `chem_xref.tsv`, which are **not** included (they are ~1.4 GB). To rebuild the
  map from scratch, download those tables from
  [MetaNetX](https://www.metanetx.org/) and run
  `python scripts/create_mapping.py`.
- Third-party comparison tools (CLIPZyme, MicrobeRX, SIMMER, TurNuP, VenusRXN)
  are **not** bundled. The scripts under `benchmarks/` that compare against them
  expect those tools to be installed separately from their own repositories.

## Citing ChemGIME



## License

Released under the [MIT License](LICENSE).
