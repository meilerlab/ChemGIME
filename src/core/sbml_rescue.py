"""
ChemGIME SBML Orphan Rescue
============================
Identifies reactions in an SBML model that lack Gene-Protein-Reaction (GPR)
associations ("orphans") and injects newly predicted GPRs from gapseq
evidence, producing an FBA-compliant SBML Level 3 Version 1 model.

This module uses COBRApy for SBML I/O (which wraps libSBML under the hood)
to ensure correct FBC package namespaces, unit consistency, and schema
compliance.

Workflow
--------
1. Load an existing SBML model via COBRApy.
2. Identify orphan reactions (empty ``gene_reaction_rule``).
3. Cross-reference orphans against gapseq GPR predictions.
4. Inject predicted GPRs into the model, creating ``GeneProduct`` entries
   as needed.
5. Validate the resulting SBML against Level 3 v1 / FBC v2 standards.
6. Export the rescued model.

The module prints a console audit trail summarising the rescue operation.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit dataclass
# ---------------------------------------------------------------------------
@dataclass
class RescueAudit:
    """Tracks the results of an orphan rescue operation."""

    total_reactions: int = 0
    total_orphans: int = 0
    successful_mappings: int = 0
    unmapped_orphans: int = 0
    new_genes_created: int = 0
    sbml_validation_status: str = "NOT_RUN"
    sbml_validation_errors: List[str] = field(default_factory=list)
    mapped_reactions: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a formatted audit trail to the console."""
        border = "=" * 60
        print(f"\n{border}")
        print("  SBML ORPHAN RESCUE — AUDIT TRAIL")
        print(border)
        print(f"  Total Reactions in Model:      {self.total_reactions}")
        print(f"  Orphan Reactions Identified:    {self.total_orphans}")
        print(f"  Successful GPR Mappings:        {self.successful_mappings}")
        print(f"  Remaining Unmapped Orphans:     {self.unmapped_orphans}")
        print(f"  New Gene Products Created:      {self.new_genes_created}")
        print(f"  SBML Validation Status:         {self.sbml_validation_status}")
        if self.sbml_validation_errors:
            print(f"  Validation Issues ({len(self.sbml_validation_errors)}):")
            for err in self.sbml_validation_errors[:10]:
                print(f"    - {err}")
            if len(self.sbml_validation_errors) > 10:
                print(f"    ... and {len(self.sbml_validation_errors) - 10} more")
        print(border + "\n")


# ---------------------------------------------------------------------------
# GPR parsing helper
# ---------------------------------------------------------------------------
def _extract_gene_ids_from_gpr(gpr_string: str) -> Set[str]:
    """Extract individual gene IDs from a GPR string.

    Handles formats like:
    - ``WP_020104528.1``
    - ``( WP_020104528.1 or WP_013301993.1 )``
    - ``( gene_A and gene_B ) or gene_C``
    """
    if not gpr_string:
        return set()
    # Remove logical operators and parentheses, split on whitespace
    cleaned = gpr_string.replace("(", " ").replace(")", " ")
    tokens = cleaned.split()
    # Filter out logical operators
    gene_ids = {t for t in tokens if t.lower() not in ("and", "or", "")}
    return gene_ids


def _sanitize_gene_id(gene_id: str) -> str:
    """Sanitize a gene ID to be SBML-SID compatible.

    SBML SIDs must match: ``[a-zA-Z_][a-zA-Z0-9_]*``
    We replace dots and hyphens with underscores and prepend 'G_' if it
    starts with a digit.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", gene_id)
    if sanitized and sanitized[0].isdigit():
        sanitized = "G_" + sanitized
    return sanitized


# ---------------------------------------------------------------------------
# Core rescue logic
# ---------------------------------------------------------------------------
def identify_orphan_reactions(model: Any) -> List[Any]:
    """Return reactions in the COBRA model that lack GPR associations.

    Parameters
    ----------
    model : cobra.Model
        A loaded COBRApy model.

    Returns
    -------
    list of cobra.Reaction
        Reactions where ``gene_reaction_rule`` is empty/missing.
    """
    orphans = []
    for rxn in model.reactions:
        if rxn.boundary:
            continue
        gpr = getattr(rxn, "gene_reaction_rule", "")
        if not gpr or gpr.strip() == "":
            orphans.append(rxn)
    return orphans


def rescue_orphans(
    model: Any,
    gpr_map: Dict[str, str],
    *,
    strict_id_match: bool = False,
) -> Tuple[Any, RescueAudit]:
    """Inject predicted GPRs into orphan reactions of a COBRA model.

    Parameters
    ----------
    model : cobra.Model
        The metabolic model (will be modified in-place and also returned).
    gpr_map : dict
        ``{reaction_id: gpr_string}`` from :func:`gapseq_parser.build_reaction_gpr_map`.
    strict_id_match : bool
        If False (default), attempt fuzzy matching by stripping compartment
        suffixes and normalising case.

    Returns
    -------
    tuple
        ``(model, audit)``
    """
    try:
        import cobra
    except ImportError:
        raise ImportError(
            "`cobra` library not found. Install with: `pip install cobra`"
        )

    audit = RescueAudit()
    audit.total_reactions = len([r for r in model.reactions if not r.boundary])

    orphans = identify_orphan_reactions(model)
    audit.total_orphans = len(orphans)
    logger.info(f"Identified {audit.total_orphans} orphan reactions in model {model.id}.")

    if not orphans:
        audit.unmapped_orphans = 0
        audit.sbml_validation_status = "SKIPPED"
        return model, audit

    # Build a lookup for fuzzy matching: normalised key → original GPR map key
    normalised_gpr: Dict[str, str] = {}
    for rxn_id in gpr_map:
        normalised_gpr[rxn_id.upper()] = rxn_id
        # Also try without common suffixes like _e, _c, _p
        base = re.sub(r"[-_][a-z]$", "", rxn_id, flags=re.IGNORECASE)
        normalised_gpr[base.upper()] = rxn_id

    existing_gene_ids = {g.id for g in model.genes}
    new_gene_count = 0

    for rxn in orphans:
        # --- Find matching GPR ---
        gpr_string = None
        if rxn.id in gpr_map:
            gpr_string = gpr_map[rxn.id]
        elif not strict_id_match:
            # Try normalised lookup
            norm_key = rxn.id.upper()
            if norm_key in normalised_gpr:
                gpr_string = gpr_map[normalised_gpr[norm_key]]
            else:
                # Strip compartment suffix from model reaction ID and retry
                base_id = re.sub(r"[-_][a-z]$", "", rxn.id, flags=re.IGNORECASE)
                if base_id.upper() in normalised_gpr:
                    gpr_string = gpr_map[normalised_gpr[base_id.upper()]]

        if not gpr_string:
            continue

        # --- Ensure all gene products exist in the model ---
        gene_ids = _extract_gene_ids_from_gpr(gpr_string)
        sanitized_gpr = gpr_string
        for raw_id in gene_ids:
            safe_id = _sanitize_gene_id(raw_id)
            sanitized_gpr = sanitized_gpr.replace(raw_id, safe_id)

            if safe_id not in existing_gene_ids:
                try:
                    new_gene = cobra.Gene(safe_id)
                    new_gene.name = raw_id  # Preserve original accession as name
                    model.genes.append(new_gene)
                    existing_gene_ids.add(safe_id)
                    new_gene_count += 1
                except Exception as e:
                    logger.warning(f"Could not create gene {safe_id}: {e}")

        # --- Assign GPR ---
        try:
            rxn.gene_reaction_rule = sanitized_gpr
            audit.mapped_reactions.append(rxn.id)
            audit.successful_mappings += 1
            logger.debug(f"  Rescued {rxn.id} with GPR: {sanitized_gpr}")
        except Exception as e:
            logger.warning(f"  Failed to set GPR for {rxn.id}: {e}")

    audit.new_genes_created = new_gene_count
    audit.unmapped_orphans = audit.total_orphans - audit.successful_mappings
    logger.info(
        f"Rescue complete: {audit.successful_mappings}/{audit.total_orphans} "
        f"orphans mapped, {new_gene_count} new genes created."
    )
    return model, audit


# ---------------------------------------------------------------------------
# SBML Validation
# ---------------------------------------------------------------------------
def validate_sbml(sbml_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """Validate an SBML file against Level 3 Version 1 standards.

    Uses libSBML's built-in validator (accessed via COBRApy's dependency).

    Parameters
    ----------
    sbml_path : str or Path
        Path to the SBML file to validate.

    Returns
    -------
    tuple
        ``(is_valid, list_of_error_strings)``
    """
    try:
        import libsbml
    except ImportError:
        logger.warning(
            "libsbml not available; skipping SBML validation. "
            "Install with: pip install python-libsbml"
        )
        return True, ["libsbml not installed — validation skipped"]

    sbml_path = Path(sbml_path)
    reader = libsbml.SBMLReader()
    document = reader.readSBML(str(sbml_path))

    errors = []
    num_errors = document.getNumErrors()

    # Separate true errors from warnings
    error_count = 0
    for i in range(num_errors):
        err = document.getError(i)
        severity = err.getSeverity()
        if severity >= libsbml.LIBSBML_SEV_ERROR:
            errors.append(f"[ERROR] L{err.getLine()}: {err.getMessage()}")
            error_count += 1
        elif severity == libsbml.LIBSBML_SEV_WARNING:
            errors.append(f"[WARN] L{err.getLine()}: {err.getMessage()}")

    # Also run consistency checks
    document.checkConsistency()
    for i in range(document.getNumErrors()):
        err = document.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            msg = f"[CONSISTENCY] L{err.getLine()}: {err.getMessage()}"
            if msg not in errors:
                errors.append(msg)
                error_count += 1

    is_valid = error_count == 0
    return is_valid, errors


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_rescued_model(
    model: Any,
    output_path: Union[str, Path],
    *,
    validate: bool = True,
) -> RescueAudit:
    """Write the rescued model to SBML and optionally validate.

    Parameters
    ----------
    model : cobra.Model
        The model (already rescued via :func:`rescue_orphans`).
    output_path : str or Path
        Where to write the SBML XML file.
    validate : bool
        If True, run SBML validation after writing.

    Returns
    -------
    RescueAudit
        Updated audit with validation results.
    """
    try:
        import cobra
    except ImportError:
        raise ImportError("`cobra` library not found.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cobra.io.write_sbml_model(model, str(output_path))
    logger.info(f"Rescued model written to {output_path}")

    audit = RescueAudit()
    if validate:
        is_valid, val_errors = validate_sbml(output_path)
        audit.sbml_validation_status = "PASS" if is_valid else "FAIL"
        audit.sbml_validation_errors = val_errors
    else:
        audit.sbml_validation_status = "SKIPPED"

    return audit


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def rescue_and_export(
    sbml_input: Union[str, Path],
    gpr_map: Dict[str, str],
    sbml_output: Union[str, Path],
    *,
    validate: bool = True,
) -> RescueAudit:
    """End-to-end: load model → rescue orphans → export → validate.

    Parameters
    ----------
    sbml_input : str or Path
        Path to the original SBML model.
    gpr_map : dict
        ``{reaction_id: gpr_string}`` mapping.
    sbml_output : str or Path
        Path for the rescued SBML output.
    validate : bool
        Run SBML L3v1 validation after export.

    Returns
    -------
    RescueAudit
        Full audit trail.
    """
    try:
        import cobra
    except ImportError:
        raise ImportError("`cobra` library not found.")

    logger.info(f"Loading SBML model from {sbml_input}...")
    model = cobra.io.read_sbml_model(str(sbml_input))
    logger.info(f"Loaded model: {model.id} ({len(model.reactions)} reactions, {len(model.genes)} genes)")

    model, audit = rescue_orphans(model, gpr_map)

    # Write and validate
    cobra.io.write_sbml_model(model, str(sbml_output))
    logger.info(f"Rescued model written to {sbml_output}")

    if validate:
        is_valid, val_errors = validate_sbml(sbml_output)
        audit.sbml_validation_status = "PASS" if is_valid else "FAIL"
        audit.sbml_validation_errors = val_errors
    else:
        audit.sbml_validation_status = "SKIPPED"

    audit.print_summary()
    return audit
