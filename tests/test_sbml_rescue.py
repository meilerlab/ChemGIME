import pytest
import cobra
from src.core.sbml_rescue import identify_orphan_reactions, _sanitize_gene_id, rescue_orphans

def test_sanitize_gene_id():
    assert _sanitize_gene_id("WP_123.1") == "WP_123_1"
    assert _sanitize_gene_id("12345.1") == "G_12345_1"
    assert _sanitize_gene_id("gene-A") == "gene_A"

def test_identify_orphan_reactions():
    model = cobra.Model("test_model")
    
    rxn1 = cobra.Reaction("R1")
    rxn1.gene_reaction_rule = "G1"
    
    rxn2 = cobra.Reaction("R2")
    rxn2.gene_reaction_rule = ""
    
    model.add_reactions([rxn1, rxn2])
    
    orphans = identify_orphan_reactions(model)
    assert len(orphans) == 1
    assert orphans[0].id == "R2"

def test_rescue_orphans():
    model = cobra.Model("test_model")
    rxn1 = cobra.Reaction("R1")
    model.add_reactions([rxn1])
    
    gpr_map = {"R1": "( WP_1.1 or WP_2.1 )", "R_not_in_model": "WP_3.1"}
    
    # Run rescue
    rescued_model, audit = rescue_orphans(model, gpr_map)
    
    assert audit.total_orphans == 1
    assert audit.successful_mappings == 1
    assert audit.new_genes_created == 2
    
    assert rescued_model.reactions.get_by_id("R1").gene_reaction_rule == "WP_1_1 or WP_2_1"
    assert "WP_1_1" in [g.id for g in rescued_model.genes]
    assert "WP_2_1" in [g.id for g in rescued_model.genes]
    
    # The original gene name should be preserved
    assert rescued_model.genes.get_by_id("WP_1_1").name == "WP_1.1"

def test_rescue_orphans_fuzzy_match():
    model = cobra.Model("test_model")
    # BiGG adds compartment suffixes sometimes
    rxn1 = cobra.Reaction("R1_c")
    model.add_reactions([rxn1])
    
    # MetaCyc mapping doesn't have suffix
    gpr_map = {"R1": "WP_1.1"}
    
    # Non-strict match should strip _c and match
    rescued_model, audit = rescue_orphans(model, gpr_map, strict_id_match=False)
    
    assert audit.successful_mappings == 1
    assert rescued_model.reactions.get_by_id("R1_c").gene_reaction_rule == "WP_1_1"
