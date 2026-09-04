# ============================================================
# ChemGIME — reproducibility Makefile
# Regenerates every benchmark table and figure from raw inputs.
# Usage:  make all            — run the full pipeline
#         make env            — create conda environment
#         make mcsa           — M-CSA cross-database benchmark
#         make loo            — M-CSA leave-one-enzyme-out benchmark
#         make split          — split-GEM retrieval benchmark
#         make cv             — stratified 5-fold cross-validation
#         make simmer         — SIMMER drug-metabolism benchmark
#         make decoy          — decoy discrimination benchmark
#         make scramble       — scrambled-reaction decoy benchmark
#         make test           — unit tests
#         make clean          — remove generated JSON / figures
# ============================================================

PYTHON   := python
GEM      := iML1515.xml
GEM_JN   := iJN746.xml
DATA     := benchmarks/data
TESTS    := tests/benchmark_output
SEED     := 42

.PHONY: all env mcsa loo split cv simmer decoy scramble test clean

all: mcsa loo split cv simmer decoy scramble

# ── Environment ─────────────────────────────────────────────
env:
	conda env create -f environment.yml
	@echo "Activate with: conda activate chemgime"

# ── M-CSA cross-database retrieval ──────────────────────────
mcsa:
	$(PYTHON) tests/benchmarking_suite.py mcsa-novelty \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-novelty \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors true  --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-novelty \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-novelty \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors true  --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-novelty \
	    --gem $(GEM_JN) --fingerprint drfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-novelty \
	    --gem $(GEM_JN) --fingerprint qrfp --filter_cofactors false --seed $(SEED)

# ── M-CSA leave-one-enzyme-out (strict held-out generalisation) ──
loo:
	$(PYTHON) tests/benchmarking_suite.py mcsa-loo \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-loo \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors true  --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-loo \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py mcsa-loo \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors true  --seed $(SEED)

# ── Split-GEM retrieval (permissive + strict GPR ground truth) ──
split:
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors true  --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors true  --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM_JN) --fingerprint drfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM_JN) --fingerprint drfp --filter_cofactors true  --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM_JN) --fingerprint qrfp --filter_cofactors false --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py split-gem \
	    --gem $(GEM_JN) --fingerprint qrfp --filter_cofactors true  --seed $(SEED)

# ── Stratified 5-fold cross-validation ──────────────────────
cv:
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors false --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM) --fingerprint drfp --filter_cofactors true  --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors false --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM) --fingerprint qrfp --filter_cofactors true  --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM_JN) --fingerprint drfp --filter_cofactors false --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM_JN) --fingerprint drfp --filter_cofactors true  --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM_JN) --fingerprint qrfp --filter_cofactors false --folds 5 --seed $(SEED)
	$(PYTHON) tests/benchmarking_suite.py cross-validate \
	    --gem $(GEM_JN) --fingerprint qrfp --filter_cofactors true  --folds 5 --seed $(SEED)

# ── SIMMER drug-metabolism benchmark ────────────────────────
simmer:
	$(PYTHON) benchmarks/simmer_vs_chemgime.py \
	    --gem $(GEM) --fp-type drfp --seed $(SEED)

# ── Decoy discrimination (cross-EC GEM) ─────────────────────
decoy:
	$(PYTHON) benchmarks/decoy_eval.py \
	    --gem $(GEM) --fp-type drfp --cofactors keep \
	    --active-def gpr \
	    --output $(DATA)/decoy_iML1515_drfp_keep_gpr.json \
	    --seed $(SEED)
	$(PYTHON) benchmarks/decoy_eval.py \
	    --gem $(GEM) --fp-type qrfp --cofactors keep \
	    --active-def gpr \
	    --output $(DATA)/decoy_iML1515_qrfp_keep.json \
	    --seed $(SEED)

# ── Scrambled-reaction decoy benchmark ──────────────────────
scramble:
	$(PYTHON) benchmarks/decoy_eval.py \
	    --gem $(GEM) --fp-type drfp --cofactors keep \
	    --scramble --n-decoys 10 \
	    --output $(DATA)/decoy_scramble_iML1515_drfp_keep.json \
	    --seed $(SEED)

# ── Unit tests ───────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -q

# ── Clean generated outputs ──────────────────────────────────
clean:
	rm -f $(DATA)/decoy_*.json $(DATA)/decoy_*_figure.png
	rm -f $(TESTS)/mcsa_novelty_*.json $(TESTS)/split_gem_*.json $(TESTS)/cv_*.json
