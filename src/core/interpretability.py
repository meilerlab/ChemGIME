"""
ChemGIME Structural Interpretability Layer
==========================================
Atom-level heat-map attribution for DRFP-based reaction similarity.

Methodology
-----------
1. Encode both the query and target reaction with ``atom_index_mapping=True``
   so that each ON bit is linked back to the RDKit atom indices that generated it.
2. Find bits that are ON in **both** fingerprints (shared / "intersecting" bits).
3. For every molecule in the query reaction, accumulate a raw score per atom:
   score[atom] += 1 for each shared bit that covered that atom.
4. Normalise scores to [0, 1].  Shift to [0.15, 1.0] so even zero-weight atoms
   receive a faint tint (avoids "data missing" impression).
5. Render each molecule with ``rdMolDraw2D.MolDraw2DSVG`` using per-atom highlight
   colours drawn from the ``YlOrRd`` (yellow → orange → red) colormap.
6. Assemble query-reactants / query-products / top-hit-reactants / top-hit-products
   into a self-contained HTML block ready for direct ``innerHTML`` injection or
   ``st.components.v1.html``.

Public API
----------
::

    from src.core.interpretability import explain_reaction_similarity, explain_top_hit

    # Tuple of (left_dot_smiles, right_dot_smiles)
    html_block = explain_reaction_similarity(
        query_smiles=("CC(=O)O", "CCO"),
        target_smiles=("CC(=O)OCC", "O"),
    )

    # Convenience wrapper for a run_single_analysis() result dict
    html_block = explain_top_hit(result)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DRFP encoding with atom-index mapping
# ---------------------------------------------------------------------------

def _encode_with_atom_map(
    left: str,
    right: str,
    radius: int = 3,
    n_bits: int = 2048,
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """Encode one reaction and return ``(fp_array, atom_maps_entry)``.

    The ``atom_maps_entry`` has the structure::

        {
            'reactants': [ {bit_pos: [frozenset_of_atom_idx, ...], ...}, ... ],
            'products':  [ {bit_pos: [frozenset_of_atom_idx, ...], ...}, ... ],
        }

    where each list element corresponds to one fragment molecule in the
    dot-separated SMILES and ``bit_pos`` is already the *folded* position
    (same integer space as ``np.where(fp_array)[0]``).

    Returns ``(None, None)`` on any failure so callers can degrade gracefully.
    """
    try:
        from drfp.fingerprint import DrfpEncoder
    except ImportError:
        logger.error("`drfp` not installed — run: pip install drfp")
        return None, None

    # Guard against empty strings, pandas NaN coerced to "nan", etc.
    _invalid = {"", "nan", "none", "null"}
    if not left or not right or left.strip().lower() in _invalid or right.strip().lower() in _invalid:
        logger.debug("_encode_with_atom_map: invalid SMILES (left=%r, right=%r)", left, right)
        return None, None

    try:
        fp_list, _result_map, atom_maps = DrfpEncoder.encode(
            f"{left}>>{right}",
            n_folded_length=n_bits,
            radius=radius,
            rings=True,
            min_radius=0,
            atom_index_mapping=True,
        )
        fp_arr = np.asarray(fp_list[0], dtype=np.uint8)
        return fp_arr, atom_maps[0]
    except Exception as exc:
        logger.warning("DRFP atom-map encoding failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Atom weight computation
# ---------------------------------------------------------------------------

def _atom_weights(
    mol_maps: List[dict],
    shared_bits: Set[int],
    n_atoms: int,
) -> np.ndarray:
    """Return a [0, 1]-normalised weight array for one molecule.

    Parameters
    ----------
    mol_maps:
        List of per-molecule bit→atom-index dicts from one side of the atom
        map (e.g. ``atom_map['reactants']``).  We try each element of the list
        so that multi-fragment dot-SMILES molecules are all covered.
    shared_bits:
        Set of folded bit positions that are ON in both query and target.
    n_atoms:
        Number of heavy atoms in the molecule (``mol.GetNumAtoms()``).
    """
    scores = np.zeros(n_atoms, dtype=float)
    for mol_map in mol_maps:
        if not isinstance(mol_map, dict):
            continue
        for bit, atom_sets in mol_map.items():
            if int(bit) not in shared_bits:
                continue
            for outer in (atom_sets or []):      # [[{0,1,2}]] → [{0,1,2}]
                for atom_set in outer:           # [{0,1,2}] → {0,1,2}
                    for idx in atom_set:         # {0,1,2} → 0,1,2
                        if 0 <= idx < n_atoms:
                            scores[idx] += 1.0

    max_s = scores.max()
    if max_s > 0:
        scores /= max_s
    return scores


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

def _weights_to_colors(
    weights: np.ndarray,
    cmap_name: str,
) -> Dict[int, Tuple[float, float, float]]:
    """Map [0, 1] weights → RGB tuples via a matplotlib colormap.

    The weights are linearly shifted to ``[0.15, 1.0]`` so every atom
    receives at least a faint tint.
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name)
        return {
            i: tuple(cmap(0.15 + 0.85 * float(w))[:3])
            for i, w in enumerate(weights)
        }
    except Exception:
        # Fallback: uniform light-yellow for every atom
        return {i: (1.0, 1.0, 0.75) for i in range(len(weights))}


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------

def _draw_mol_svg(
    mol,
    weights: np.ndarray,
    width: int,
    height: int,
    cmap_name: str,
    min_weight: float,
) -> str:
    """Draw *mol* to an SVG string with per-atom heat-map highlighting.

    Atoms with ``weight < min_weight`` are excluded from the highlight list
    so the drawing does not look uniformly coloured when similarity is low.
    """
    try:
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        rdDepictor.Compute2DCoords(mol)
        atom_colors = _weights_to_colors(weights, cmap_name)

        highlight_atoms = [i for i, w in enumerate(weights) if w >= min_weight]
        highlight_colors = {i: atom_colors[i] for i in highlight_atoms}

        # Colour bonds between two highlighted atoms
        highlight_bonds: List[int] = []
        bond_colors: Dict[int, Tuple[float, float, float]] = {}
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in highlight_colors and a2 in highlight_colors:
                w_avg = (weights[a1] + weights[a2]) / 2.0
                bond_colors[bond.GetIdx()] = tuple(
                    _weights_to_colors(np.array([w_avg]), cmap_name)[0]
                )
                highlight_bonds.append(bond.GetIdx())

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = False
        opts.addAtomIndices = False
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        logger.debug("SVG draw failed: %s", exc)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="#f5f5f5"/>'
            f'<text x="10" y="20" font-size="12" fill="#999">draw error</text>'
            f'</svg>'
        )


# ---------------------------------------------------------------------------
# HTML panel assembly
# ---------------------------------------------------------------------------

def _parse_dot_smiles(dot_smiles: str) -> List[Tuple[str, object]]:
    """Split a dot-separated SMILES string into ``[(smi, RDKit_mol), ...]``.

    Fragments that cannot be parsed are silently skipped.
    """
    try:
        from rdkit.Chem import MolFromSmiles
        result = []
        for smi in dot_smiles.split("."):
            smi = smi.strip()
            if not smi:
                continue
            mol = MolFromSmiles(smi)
            if mol is not None:
                result.append((smi, mol))
        return result
    except Exception:
        return []


_CSS = """
<style>
.cg-attr-wrap{font-family:Arial,sans-serif;font-size:13px;color:#333;}
.cg-attr-wrap h3{margin:0 0 8px 0;font-size:15px;border-bottom:1px solid #e0e0e0;padding-bottom:4px;}
.cg-attr-side{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start;margin-bottom:8px;}
.cg-attr-mol{text-align:center;}
.cg-attr-mol span{display:block;font-size:11px;color:#666;margin-top:2px;}
.cg-attr-panel{background:#fafafa;border:1px solid #e8e8e8;border-radius:6px;padding:10px 12px;margin-bottom:10px;}
.cg-attr-panel h4{margin:0 0 6px 0;font-size:13px;color:#555;}
.cg-legend{display:flex;align-items:center;gap:8px;margin:6px 0 4px 0;font-size:11px;color:#666;}
.cg-legend-bar{width:100px;height:10px;border-radius:3px;
  background:linear-gradient(to right,#ffffb2,#fecc5c,#fd8d3c,#e31a1c);
  border:1px solid #ccc;}
.cg-stats{font-size:11px;color:#888;margin-bottom:6px;}
</style>
"""


def _reaction_panels_html(
    left_dot: str,
    right_dot: str,
    left_maps: List[dict],
    right_maps: List[dict],
    shared_bits: Set[int],
    label: str,
    mol_width: int,
    mol_height: int,
    cmap_name: str,
    min_weight: float,
) -> str:
    """Return the HTML for one reaction (reactants + products panels)."""

    def _side_html(dot_smi: str, side_maps: List[dict], side_label: str) -> str:
        mols = _parse_dot_smiles(dot_smi)
        if not mols:
            return ""
        items = []
        for mol_idx, (smi, mol) in enumerate(mols):
            weights = _atom_weights(
                side_maps[mol_idx : mol_idx + 1],
                shared_bits,
                mol.GetNumAtoms(),
            )
            svg = _draw_mol_svg(mol, weights, mol_width, mol_height, cmap_name, min_weight)
            items.append(
                f'<div class="cg-attr-mol">{svg}'
                f'<span>mol {mol_idx + 1}</span></div>'
            )
        return (
            f'<div class="cg-attr-panel">'
            f'<h4>{label} — {side_label}</h4>'
            f'<div class="cg-attr-side">{"".join(items)}</div>'
            f'</div>'
        )

    react_html = _side_html(left_dot, left_maps, "Reactants")
    prod_html = _side_html(right_dot, right_maps, "Products")
    return react_html + prod_html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_reaction_similarity(
    query_smiles: Tuple[str, str],
    target_smiles: Tuple[str, str],
    *,
    mol_width: int = 280,
    mol_height: int = 200,
    min_weight: float = 0.05,
    cmap_name: str = "YlOrRd",
    title: str = "Structural Attribution — Why is this the top hit?",
    radius: int = 3,
    n_bits: int = 2048,
) -> Optional[str]:
    """Generate a self-contained HTML heat-map showing atomic attribution.

    Highlighted atoms are those whose DRFP circular substructure shingles
    appear in **both** the query and the top-hit reaction fingerprint.
    Colour intensity (yellow → red) reflects how many shared bits covered
    each atom.

    Parameters
    ----------
    query_smiles:
        ``(left_dot_smiles, right_dot_smiles)`` for the query reaction.
    target_smiles:
        ``(left_dot_smiles, right_dot_smiles)`` for the target reaction.
    mol_width, mol_height:
        Per-molecule SVG canvas dimensions in pixels.
    min_weight:
        Atoms with normalised weight below this threshold are not highlighted.
    cmap_name:
        Matplotlib colormap (default ``"YlOrRd"``).
    title:
        Heading rendered at the top of the block.
    radius, n_bits:
        DRFP parameters — must match those used for the similarity search.

    Returns
    -------
    str or None
        Self-contained HTML string, or ``None`` when attribution cannot be
        computed (missing library, SMILES parse failure, no shared bits, …).
    """
    q_left, q_right = query_smiles
    t_left, t_right = target_smiles

    q_fp, q_map = _encode_with_atom_map(q_left, q_right, radius=radius, n_bits=n_bits)
    t_fp, t_map = _encode_with_atom_map(t_left, t_right, radius=radius, n_bits=n_bits)

    if q_fp is None or t_fp is None:
        logger.debug("Attribution skipped: fingerprint encoding failed.")
        return None

    shared_bits: Set[int] = {int(b) for b in np.where(q_fp & t_fp)[0]}
    if not shared_bits:
        logger.debug("Attribution skipped: no shared bits.")
        return None

    n_shared = len(shared_bits)
    n_union = int((q_fp | t_fp).sum())
    jaccard = n_shared / n_union if n_union > 0 else 0.0

    q_react_maps = (q_map or {}).get("reactants", [])
    q_prod_maps = (q_map or {}).get("products", [])
    t_react_maps = (t_map or {}).get("reactants", [])
    t_prod_maps = (t_map or {}).get("products", [])

    query_panels = _reaction_panels_html(
        q_left, q_right, q_react_maps, q_prod_maps,
        shared_bits, "Query", mol_width, mol_height, cmap_name, min_weight,
    )
    target_panels = _reaction_panels_html(
        t_left, t_right, t_react_maps, t_prod_maps,
        shared_bits, "Top Hit", mol_width, mol_height, cmap_name, min_weight,
    )

    legend = (
        '<div class="cg-legend">'
        '<span>Low contribution</span>'
        '<div class="cg-legend-bar"></div>'
        '<span>High contribution</span>'
        '</div>'
    )
    stats = (
        f'<div class="cg-stats">'
        f'Shared DRFP bits: <strong>{n_shared}</strong> &nbsp;|&nbsp; '
        f'Jaccard similarity: <strong>{jaccard:.3f}</strong> &nbsp;|&nbsp; '
        f'Radius: {radius}, Bits: {n_bits}'
        f'</div>'
    )

    html = (
        f'{_CSS}'
        f'<div class="cg-attr-wrap">'
        f'<h3>{title}</h3>'
        f'{stats}'
        f'{legend}'
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;">'
        f'<div style="flex:1;min-width:260px;">{query_panels}</div>'
        f'<div style="flex:1;min-width:260px;">{target_panels}</div>'
        f'</div>'
        f'<p style="font-size:10px;color:#aaa;margin:4px 0 0 0;">'
        f'Atoms coloured by the number of matching DRFP circular substructure '
        f'shingles (Morgan-style, radius {radius}) that they contributed to '
        f'the shared fingerprint bits.'
        f'</p>'
        f'</div>'
    )
    return html


def explain_top_hit(
    result: dict,
    *,
    title: str = "Structural Attribution — Why is this the top hit?",
    **kw,
) -> Optional[str]:
    """Convenience wrapper for a ``run_single_analysis()`` result dict.

    Extracts SMILES for the query reaction and the #1-ranked GEM reaction,
    then delegates to :func:`explain_reaction_similarity`.

    The result dict is expected to contain:

    * ``"query_df"`` — ``pd.DataFrame`` with columns
      ``left_smiles`` / ``right_smiles``
    * ``"df_rank"`` — ``pd.DataFrame`` with columns
      ``raw_left_smiles`` / ``raw_right_smiles``, sorted by similarity

    Returns ``None`` on any failure so callers can degrade gracefully.
    """
    try:
        query_df = result.get("query_df")
        df_rank = result.get("df_rank")
        if query_df is None or df_rank is None or df_rank.empty:
            return None

        q_row = query_df.iloc[0]
        t_row = df_rank.iloc[0]

        q_left = str(q_row.get("left_smiles") or q_row.get("raw_left_smiles") or "")
        q_right = str(q_row.get("right_smiles") or q_row.get("raw_right_smiles") or "")
        t_left = str(t_row.get("raw_left_smiles") or "")
        t_right = str(t_row.get("raw_right_smiles") or "")

        if not all([q_left, q_right, t_left, t_right]):
            logger.debug("explain_top_hit: SMILES missing from result dict.")
            return None

        return explain_reaction_similarity(
            (q_left, q_right),
            (t_left, t_right),
            title=title,
            **kw,
        )
    except Exception as exc:
        logger.debug("explain_top_hit failed: %s", exc)
        return None
