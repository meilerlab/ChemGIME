import pytest
import pandas as pd
from pathlib import Path
from src.core.gapseq_parser import _extract_protein_id, parse_gapseq_tbl, build_gene_reaction_map, build_scoring_features

def test_extract_protein_id():
    assert _extract_protein_id("WP_020104528.1 dihydrolipoyl dehydrogenase") == "WP_020104528.1"
    assert _extract_protein_id("NP_12345.1 something") == "NP_12345.1"
    assert _extract_protein_id("unrelated text") is None
    assert _extract_protein_id(None) is None

def test_parse_gapseq_tbl(tmp_path):
    tbl_content = (
        "# Gapseq tbl file\n"
        "rxn\tname\tec\tbihit\tqseqid\tpident\tevalue\tbitscore\tqcovs\tstitle\tsstart\tsend\tpathway\tstatus\tpathway.status\tdbhit\tcomplex\texception\tcomplex.status\n"
        "RXN-1\tSome Reaction\t1.1.1.1\tTrue\tQ123\t95.5\t1e-50\t200\t100\tWP_020104528.1 xyz\t1\t100\tpathA\tgood_blast\tp_stat\tdbh\tcpx\texc\tcpx_stat\n"
        "RXN-2\tAnother Rxn\t2.2.2.2\tFalse\tQ456\t20.0\t1.0\t40\t30\tbad WP_99999.1\t1\t100\tpathB\tbad_blast\tp_stat\tdbh\tcpx\texc\tcpx_stat\n"
    )
    tbl_file = tmp_path / "test.tbl"
    tbl_file.write_text(tbl_content)
    
    df = parse_gapseq_tbl(tbl_file, status_filter=["good_blast"])
    assert len(df) == 1
    assert df.iloc[0]["rxn"] == "RXN-1"
    assert df.iloc[0]["protein_id"] == "WP_020104528.1"
    
    # Test numeric parsing
    assert df.iloc[0]["pident"] == 95.5
    assert df.iloc[0]["evalue"] == pytest.approx(1e-50)

def test_build_gene_reaction_map():
    df = pd.DataFrame({
        "rxn": ["RXN-1", "RXN-2", "RXN-1"],
        "protein_id": ["WP_1.1", "WP_1.1", "WP_2.1"],
        "pident": [90.0, 95.0, 80.0],
        "evalue": [1e-10, 1e-20, 1e-5],
        "ec": ["1.1.1.1", "2.2.2.2", "1.1.1.1/something"]
    })
    
    gmap = build_gene_reaction_map(df)
    assert len(gmap) == 2
    assert "WP_1.1" in gmap
    assert "WP_2.1" in gmap
    
    assert gmap["WP_1.1"].reaction_ids == {"RXN-1", "RXN-2"}
    assert gmap["WP_1.1"].best_pident == 95.0
    assert gmap["WP_1.1"].best_evalue == 1e-20
    assert "2.2.2.2" in gmap["WP_1.1"].ec_numbers
    
def test_build_scoring_features():
    df = pd.DataFrame({
        "rxn": ["RXN-1", "RXN-1"],
        "protein_id": ["WP_1.1", "WP_1.1"],
        "name": ["Rxn 1", "Rxn 1"],
        "ec": ["1.1.1.1", "1.1.1.1"],
        "qseqid": ["Q1", "Q2"],
        "pident": [90.0, 95.0],
        "evalue": [1e-10, 1e-20],
        "qcovs": [80.0, 100.0],
        "bitscore": [100.0, 150.0],
        "stitle": ["st 1", "st 2"]
    })
    
    agg = build_scoring_features(df)
    assert len(agg) == 1
    assert agg.iloc[0]["rxn"] == "RXN-1"
    assert agg.iloc[0]["protein_id"] == "WP_1.1"
    assert agg.iloc[0]["best_pident"] == 95.0
    assert agg.iloc[0]["best_evalue"] == 1e-20
    assert agg.iloc[0]["best_qcovs"] == 100.0
    assert agg.iloc[0]["hit_count"] == 2
