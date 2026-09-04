"""
ChemGIME Reporting
==================
HTML report generation and molecular/reaction image rendering.
"""

import base64
import logging
from datetime import datetime
from typing import Optional, Set

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

from .similarity import find_most_similar_reactant, get_all_reactants_from_smiles

logger = logging.getLogger(__name__)


def run_rxn(smiles_l: str, smiles_r: str) -> Optional[rdChemReactions.ChemicalReaction]:
    """Generates an RDKit ChemicalReaction object from SMILES (for drawing)."""
    try:
        sms_l = str(smiles_l).split('.') if pd.notna(smiles_l) else []
        sms_r = str(smiles_r).split('.') if pd.notna(smiles_r) else []
        if not sms_l and not sms_r:
            return None
        smas_l = [Chem.MolToSmarts(Chem.MolFromSmiles(sm.replace('R', '*')), isomericSmiles=True) for sm in sms_l if Chem.MolFromSmiles(sm.replace('R', '*'))]
        smas_r = [Chem.MolToSmarts(Chem.MolFromSmiles(sm.replace('R', '*')), isomericSmiles=True) for sm in sms_r if Chem.MolFromSmiles(sm.replace('R', '*'))]
        return rdChemReactions.ReactionFromSmarts(f"{'.'.join(smas_l)}>>{'.'.join(smas_r)}")
    except Exception:
        return None


def render_img_to_base64(smiles, is_reaction=False) -> str:
    """Renders a molecule or reaction and returns a base64 string for HTML embedding."""
    if not smiles or (isinstance(smiles, tuple) and (pd.isna(smiles[0]) or pd.isna(smiles[1]))):
        return ""
    try:
        drawer = None
        if is_reaction:
            smiles_l, smiles_r = smiles
            mol_or_rxn = run_rxn(smiles_l, smiles_r)
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 400)
            if mol_or_rxn:
                drawer.DrawReaction(mol_or_rxn)
        else:
            mol_or_rxn = Chem.MolFromSmiles(smiles)
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 250)
            drawer.drawOptions().addStereoAnnotation = True
            if mol_or_rxn:
                drawer.DrawMolecule(mol_or_rxn)

        if not mol_or_rxn:
            return ""
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        base64_str = base64.b64encode(png_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    except Exception:
        return ""


def generate_html_report(
    df_rank: pd.DataFrame,
    query_df: pd.DataFrame,
    gem_filename: str,
    query_filename: str,
    num_results: int,
    ec_predictions: dict,
    cofactor_set: Set[str]
):
    """
    Generates the HTML report, including GPRs.
    """
    query_row = query_df.iloc[0]

    all_query_reactants = get_all_reactants_from_smiles(
        query_row.get('left_smiles'),
        cofactor_set
    )

    df_top_reactions = df_rank.head(num_results).copy()
    df_top_reactions.insert(0, 'Rank', range(1, len(df_top_reactions) + 1))

    display_cols_rxn = ['Rank', 'reaction_id', 'name', 'gpr', 'ec', 'subsystem', 'tan_distance']
    display_cols_rxn = [col for col in display_cols_rxn if col in df_top_reactions.columns]

    # --- Extract EC prediction results ---
    l1_pred, l1_conf, l1_df = ec_predictions["L1"]
    l2_pred, l2_conf, l2_df = ec_predictions["L2"]
    l3_pred, l3_conf, l3_df = ec_predictions["L3"]
    l4_pred, l4_conf, l4_df = ec_predictions["L4"]

    # --- Start HTML Generation ---
    html = f"""
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Reaction Similarity Report</title>
<style>
body {{font-family: sans-serif; margin: 2em; color: #333; background-color: #FFF;}} h1, h2, h3 {{border-bottom: 2px solid #ddd; padding-bottom: 5px;}} table {{border-collapse: collapse; width: 100%; font-size: 0.9em; table-layout: fixed;}} th, td {{border: 1px solid #ddd; padding: 8px; text-align: left; overflow-wrap: break-word;}} th {{background-color: #f2f2f2;}} .code {{font-family: monospace; background: #eee; padding: 2px 4px; border-radius: 3px;}} img {{max-width: 100%; height: auto; border: 1px solid #eee;}} .table-container {{max-height: 450px; overflow-y: auto; margin-bottom: 2em; border: 1px solid #ddd;}} .grid-container {{display: grid; grid-template-columns: 1.5fr 1fr 1fr 0.8fr; gap: 1em; align-items: center; border-bottom: 1px solid #eee; padding: 1em 0;}} .grid-header {{font-weight: bold; border-bottom: 2px solid #ccc;}}
/* Styles for GPR column */
td:nth-child(4), th:nth-child(4) {{ font-family: monospace; font-size: 0.85em; }}
/* EC Prediction Box */
.ec-summary-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5em; background: #f9f9f9; border: 1px solid #ddd; padding: 1.5em; border-radius: 8px; margin-bottom: 2em;}}
.ec-level-box {{border: 1px solid #eee; border-radius: 5px; padding: 0 1em 1em 1em; background: #fff; display: flex; flex-direction: column;}}
.ec-level-box .table-container {{ max-height: 250px; margin-top: 1em; border: 1px solid #ddd; flex-grow: 1;}}
.ec-level-box h3 {{ font-size: 1.2em; margin-bottom: 0; border: none; }}
.ec-level-box h4 {{ font-size: 1.0em; margin-top: 5px; color: #555; }}
.ec-level-box p {{ font-size: 0.9em; margin-bottom: 5px;}}
</style></head>
<body>
<h1>Enzyme Identification Report</h1><p><strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<p><strong>Organism GEM:</strong> {gem_filename}</p>
<p><strong>Query File:</strong> {query_filename}</p>
<h2>Query Reaction</h2><p><strong>ID:</strong> <span class="code">{query_row.get('id', 'N/A')}</span></p>
<img src="{render_img_to_base64((query_row.get('left_smiles'), query_row.get('right_smiles')), is_reaction=True)}" alt="Query Reaction">

<h2>Predicted EC Numbers (Similarity-Weighted Vote)</h2>
<p style="margin-top: -10px; font-size: 0.9em;">Prediction based on a weighted vote of similar reactions within the GEM. Higher confidence indicates stronger agreement among top matches.</p>
<div class="ec-summary-grid">
    <div class="ec-level-box"><h3>L1: <span class="code">{l1_pred}</span></h3><h4>Confidence: <strong>{l1_conf:.2%}</strong></h4><p><strong>Top Votes (Class):</strong></p><div class="table-container">{l1_df.to_html(index=False, classes='table', na_rep='-')}</div></div>
    <div class="ec-level-box"><h3>L2: <span class="code">{l2_pred}</span></h3><h4>Confidence: <strong>{l2_conf:.2%}</strong></h4><p><strong>Top Votes (Subclass):</strong></p><div class="table-container">{l2_df.to_html(index=False, classes='table', na_rep='-')}</div></div>
    <div class="ec-level-box"><h3>L3: <span class="code">{l3_pred}</span></h3><h4>Confidence: <strong>{l3_conf:.2%}</strong></h4><p><strong>Top Votes (Sub-subclass):</strong></p><div class="table-container">{l3_df.to_html(index=False, classes='table', na_rep='-')}</div></div>
    <div class="ec-level-box"><h3>L4: <span class="code">{l4_pred}</span></h3><h4>Confidence: <strong>{l4_conf:.2%}</strong></h4><p><strong>Top Votes (Serial):</strong></p><div class="table-container">{l4_df.to_html(index=False, classes='table', na_rep='-')}</div></div>
</div>

<h2>1. Top Similar Reactions &amp; Associated Genes (GPR)</h2>
<p>Based on chemical transformation similarity (DRFP). Lower distance is more similar. Table is scrollable.</p>
<div class="table-container">
{df_top_reactions[display_cols_rxn].rename(columns={'tan_distance': 'Tanimoto Distance', 'reaction_id': 'GEM Reaction ID', 'gpr': 'Gene-Protein-Rule (GPR)', 'ec': 'EC Number', 'subsystem': 'Subsystem'}).to_html(index=False, classes='table', na_rep='-')}
</div>
"""

    # --- Structural Attribution Section (top-hit atom-level heat-map) ---
    _attr_error: str = ""
    attr_html: Optional[str] = None
    try:
        from .interpretability import explain_reaction_similarity

        def _safe_smi(val) -> str:
            """Return a clean SMILES string or '' — guards against pandas NaN."""
            if val is None:
                return ""
            try:
                import math
                if isinstance(val, float) and math.isnan(val):
                    return ""
            except Exception:
                pass
            s = str(val).strip()
            return "" if s.lower() in ("nan", "none", "") else s

        top_hit = df_rank.iloc[0] if not df_rank.empty else None
        if top_hit is not None:
            _q_left = _safe_smi(query_row.get("left_smiles")) or _safe_smi(query_row.get("raw_left_smiles"))
            _q_right = _safe_smi(query_row.get("right_smiles")) or _safe_smi(query_row.get("raw_right_smiles"))
            _t_left = _safe_smi(top_hit.get("raw_left_smiles"))
            _t_right = _safe_smi(top_hit.get("raw_right_smiles"))

            if all([_q_left, _q_right, _t_left, _t_right]):
                attr_html = explain_reaction_similarity(
                    (_q_left, _q_right), (_t_left, _t_right)
                )
                if attr_html is None:
                    _attr_error = "No shared DRFP bits found between query and top hit."
            else:
                missing = [k for k, v in {
                    "query left": _q_left, "query right": _q_right,
                    "hit left": _t_left, "hit right": _t_right,
                }.items() if not v]
                _attr_error = f"Missing SMILES: {', '.join(missing)}."
    except Exception as _exc:
        logger.warning("Structural attribution failed: %s", _exc, exc_info=True)
        _attr_error = str(_exc)

    if attr_html:
        html += f"""
<h2>3. Structural Attribution</h2>
<p>Atom-level heat-map showing which parts of the query and top-ranked GEM reaction share
matching DRFP circular substructure shingles. Colour intensity (yellow&rarr;red) reflects
how many shared fingerprint bits covered each atom. Only the #1-ranked reaction is shown.</p>
<details style="border:1px solid #ddd;border-radius:6px;padding:12px 16px;margin-bottom:2em;background:#fafafa;">
<summary style="cursor:pointer;font-weight:bold;font-size:0.95em;color:#333;">
&#9654; Click to expand atom-level attribution (query vs top hit)
</summary>
<div style="margin-top:12px;">{attr_html}</div>
</details>
"""
    else:
        html += f"""
<h2>3. Structural Attribution</h2>
<p style="color:#888;font-style:italic;">Attribution unavailable.
{"Reason: " + _attr_error if _attr_error else ""}
</p>
"""

    # --- Substrate Analysis Section ---
    if not all_query_reactants:
        html += "<h2>2. Substrate Similarity Analysis</h2><p>No valid, non-cofactor query substrates could be extracted from the query's 'left_smiles' field.</p>"
    else:
        for i, (query_smiles, query_mol) in enumerate(all_query_reactants):
            df_substrate_analysis = df_top_reactions.copy()
            try:
                query_reactant_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
                substrate_results = [
                    find_most_similar_reactant(query_reactant_fp, smi, cofactor_set)
                    for smi in df_substrate_analysis['raw_left_smiles']
                ]
                sims, compounds = zip(*substrate_results)
                df_substrate_analysis['Substrate_Tanimoto'] = sims
                df_substrate_analysis['Most_Similar_Substrate_SMILES'] = compounds
            except Exception as e:
                logger.warning(f"Could not process substrate {query_smiles}: {e}")
                df_substrate_analysis['Substrate_Tanimoto'] = 0.0
                df_substrate_analysis['Most_Similar_Substrate_SMILES'] = "Error"

            df_substrate = df_substrate_analysis.sort_values(by='Substrate_Tanimoto', ascending=False)
            main_query_img_src = render_img_to_base64(query_smiles)

            html += f"""
            <h2>2. Substrate Similarity Analysis (Query Substrate {i+1}: <span class="code">{query_smiles}</span>)</h2>
            <p>Side-by-side comparison of query substrate {i+1} to the substrates of the top reactions, sorted by compound similarity (higher score is more similar).</p>
            <div class="table-container">
            <div class="grid-container grid-header"><div>Reaction Info</div><div>Query Substrate</div><div>Database Substrate</div><div>Similarity</div></div>
            """

            for _, row in df_substrate.iterrows():
                html += f"""
                <div class="grid-container">
                    <div>
                        <strong>Name:</strong> {row.get('name', 'N/A')}<br>
                        <strong>EC:</strong> <span class="code">{row.get('ec', 'N/A')}</span><br>
                        <strong>Reaction ID:</strong> <span class="code">{row.get('reaction_id', 'N/A')}</span><br>
                        <i>(Reaction Rank: {row.get('Rank', 'N/A')})</i>
                    </div>
                    <div><img src="{main_query_img_src}" alt="Query Substrate: {query_smiles}"></div>
                    <div><img src="{render_img_to_base64(row.get('Most_Similar_Substrate_SMILES'))}" alt="Database Substrate"></div>
                    <div><center><h3>{row.get('Substrate_Tanimoto', 0.0):.4f}</h3></center></div>
                </div>"""

            html += "</div>"
    html += "</div></body></html>"
    return html
