"""
ChemGIME Core
=============
Public API for the ChemGIME chemistry engine.

This package re-exports all public functions so consumers can do:
    from src.core import build_organism_database, predict_ec_weighted, ...
"""

# --- Configuration ---
from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    DB_PATH,
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
    BIGG_RXN_SMILES_FILE,
    MODELSEED_RXN_SMILES_FILE,   # legacy alias → BIGG_RXN_SMILES_FILE
    MAX_WORKERS,
    REPORTS_DIR,
    _bootstrap_bigg_rxn_smiles,
)

# Ensure the reaction-SMILES lookup exists on first import.
_bootstrap_bigg_rxn_smiles()

# --- Data Loading ---
from .data_loading import (
    load_cofactor_set,
    load_bigg_to_smiles_map,
    load_rxn_smiles_map,
    load_data,
)

# --- GEM Parsing ---
from .gem_parser import (
    filter_smiles_string,
    build_smiles_from_reaction,
    build_organism_database,
)

# --- Gapseq Parser ---
from .gapseq_parser import (
    parse_gapseq_tbl,
    build_gene_reaction_map,
    build_reaction_gpr_map,
    build_scoring_features,
    load_gapseq_evidence,
    build_organism_database_from_tbl,
)

# --- Fingerprints ---
from .fingerprints import (
    FingerprintType,
    _calculate_drfp_fingerprint,
    _calculate_drfp_substrate_fingerprint,
    _calculate_qrfp_fingerprint,
    calculate_fingerprints_parallel,
)

# --- Similarity ---
from .similarity import (
    get_closest_rxns,
    get_all_reactants_from_smiles,
    find_most_similar_reactant,
)

# --- Reaction Center Scoring ---
from .reaction_center import (
    extract_reaction_center,
    context_aware_tanimoto,
    batch_context_scores,
)

# --- EC Prediction ---
from .ec_prediction import (
    _calculate_prediction_from_scores,
    predict_ec_weighted,
)

# --- SBML Orphan Rescue ---
from .sbml_rescue import (
    identify_orphan_reactions,
    rescue_orphans,
    validate_sbml,
    rescue_and_export,
    RescueAudit,
)

# --- Reporting ---
from .reporting import (
    run_rxn,
    render_img_to_base64,
    generate_html_report,
)

# --- Metrics (F10) ---
from .metrics import (
    reciprocal_rank,
    mean_reciprocal_rank,
    top_k_accuracy,
    retrieval_summary,
)

# --- Specificity / Decoy-Discrimination Metrics ---
from .specificity import (
    roc_auc,
    roc_points,
    average_precision,
    enrichment_factor,
    bedroc,
    precision_at_k,
    recall_at_k,
    rates_at_threshold,
    discrimination_summary,
    bootstrap_ci,
)

# --- Structured Export (F5) ---
from .export import (
    export_results,
    export_batch_summary,
)

# --- Visualization (F7) ---
from .visualization import (
    tanimoto_distance_plot,
    distance_histogram,
    ec_confidence_radar,
)

# --- Confidence Calibration (F6) ---
from .confidence import (
    compute_confidence,
    calibrate_confidence,
    fit_calibration_model,
)

# --- Batch Processing (F4) ---
from .batch import run_batch

# --- Structural Interpretability ---
from .interpretability import (
    explain_reaction_similarity,
    explain_top_hit,
)
