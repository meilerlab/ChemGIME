"""
ChemGIME Interactive Visualization (F7)
========================================
Interactive distance plots for identifying "jump out" hits —
reactions that score anomalously well against the query.

Uses Plotly for self-contained HTML output (no server dependency).

Usage::

    from src.core.visualization import tanimoto_distance_plot

    html_div = tanimoto_distance_plot(df_rank, top_n=100, highlight_n=5)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Plotly is optional — degrade gracefully
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.info("Plotly not installed — interactive plots disabled. Install with: pip install plotly")


def tanimoto_distance_plot(
    df_rank: pd.DataFrame,
    *,
    top_n: int = 100,
    highlight_n: int = 10,
    title: str = "Tanimoto Distance Landscape",
    height: int = 500,
) -> str:
    """Generate an interactive Plotly scatter plot of Tanimoto distances.

    Reactions are plotted by rank (x-axis) vs. Tanimoto distance (y-axis).
    The top ``highlight_n`` hits are coloured differently and annotated.

    The "jump out" pattern appears as a sharp elbow in the distance curve,
    where a few reactions are much more similar than the rest.

    Parameters
    ----------
    df_rank : pd.DataFrame
        Ranked reactions (sorted by tan_distance ascending).
    top_n : int
        Number of reactions to include in the plot.
    highlight_n : int
        Number of top hits to highlight in a different colour.
    title : str
        Plot title.
    height : int
        Plot height in pixels.

    Returns
    -------
    str
        Self-contained HTML div string (embeddable in reports).
        Returns a static fallback message if Plotly is not installed.
    """
    if not HAS_PLOTLY:
        return _static_fallback(df_rank, top_n, highlight_n)

    df = df_rank.head(top_n).copy()
    df["rank"] = range(1, len(df) + 1)

    # Prepare hover text
    df["hover"] = df.apply(
        lambda r: (
            f"<b>Rank {r['rank']}</b><br>"
            f"Reaction: {r.get('reaction_id', 'N/A')}<br>"
            f"Name: {r.get('name', 'N/A')}<br>"
            f"EC: {r.get('ec', 'N/A')}<br>"
            f"GPR: {str(r.get('gpr', 'N/A'))[:60]}<br>"
            f"Distance: {r.get('tan_distance', 0):.4f}"
        ),
        axis=1,
    )

    # Split into highlighted and background
    df_top = df.head(highlight_n)
    df_rest = df.iloc[highlight_n:]

    fig = go.Figure()

    # Background points
    if not df_rest.empty:
        fig.add_trace(go.Scatter(
            x=df_rest["rank"],
            y=df_rest["tan_distance"],
            mode="markers",
            marker=dict(size=6, color="#B0BEC5", opacity=0.6),
            text=df_rest["hover"],
            hoverinfo="text",
            name="Other reactions",
        ))

    # Highlighted top hits
    fig.add_trace(go.Scatter(
        x=df_top["rank"],
        y=df_top["tan_distance"],
        mode="markers+text",
        marker=dict(
            size=12,
            color=df_top["tan_distance"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Distance"),
            line=dict(width=1, color="black"),
        ),
        text=[f"#{r}" for r in df_top["rank"]],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=df_top["hover"],
        hoverinfo="text",
        name=f"Top {highlight_n} hits",
    ))

    # Elbow detection: find the rank where the distance gradient changes most
    if len(df) > 3:
        dists = df["tan_distance"].values
        gradient = np.diff(dists)
        if len(gradient) > 1:
            elbow_idx = int(np.argmax(gradient)) + 1
            elbow_dist = dists[elbow_idx]
            fig.add_hline(
                y=elbow_dist,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Elbow (rank {elbow_idx + 1})",
                annotation_position="top right",
            )

    fig.update_layout(
        title=title,
        xaxis_title="Rank",
        yaxis_title="Tanimoto Distance",
        height=height,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="closest",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def distance_histogram(
    df_rank: pd.DataFrame,
    *,
    bins: int = 50,
    title: str = "Tanimoto Distance Distribution",
    height: int = 400,
) -> str:
    """Generate a histogram of Tanimoto distances with highlighted outliers.

    Parameters
    ----------
    df_rank : pd.DataFrame
        Ranked reactions.
    bins : int
        Number of histogram bins.
    title, height : str, int
        Plot configuration.

    Returns
    -------
    str
        HTML div string.
    """
    if not HAS_PLOTLY:
        return "<p><em>Install plotly for interactive histograms.</em></p>"

    distances = df_rank["tan_distance"].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=distances,
        nbinsx=bins,
        marker_color="#42A5F5",
        opacity=0.8,
        name="All reactions",
    ))

    # Mark the top 5% (low distance = high similarity)
    threshold_5pct = distances.quantile(0.05)
    fig.add_vline(
        x=threshold_5pct,
        line_dash="dash",
        line_color="red",
        annotation_text=f"5th percentile ({threshold_5pct:.3f})",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Tanimoto Distance",
        yaxis_title="Count",
        height=height,
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def ec_confidence_radar(
    ec_predictions: dict,
    *,
    title: str = "EC Prediction Confidence",
    height: int = 400,
) -> str:
    """Radar chart showing EC prediction confidence across levels.

    Parameters
    ----------
    ec_predictions : dict
        Output of predict_ec_weighted.

    Returns
    -------
    str
        HTML div string.
    """
    if not HAS_PLOTLY:
        return "<p><em>Install plotly for interactive charts.</em></p>"

    levels = ["L1", "L2", "L3", "L4"]
    confs = []
    labels = []
    for lvl in levels:
        if lvl in ec_predictions:
            pred, conf, _ = ec_predictions[lvl]
            confs.append(conf)
            labels.append(f"{lvl}: {pred}")
        else:
            confs.append(0)
            labels.append(f"{lvl}: N/A")

    fig = go.Figure(go.Scatterpolar(
        r=confs + [confs[0]],  # close the polygon
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(66, 165, 245, 0.3)",
        line=dict(color="#1E88E5"),
    ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
        height=height,
        template="plotly_white",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _static_fallback(df_rank: pd.DataFrame, top_n: int, highlight_n: int) -> str:
    """ASCII-art fallback when Plotly is unavailable."""
    df = df_rank.head(highlight_n)
    rows = []
    for _, r in df.iterrows():
        dist = r.get("tan_distance", 0)
        bar = "#" * int(dist * 40)
        rxn = r.get("reaction_id", "?")
        rows.append(f"  {rxn:20s} | {dist:.4f} |{bar}")

    header = f"<pre>Top {highlight_n} reactions by Tanimoto distance:\n"
    header += f"  {'Reaction ID':20s} | {'Dist':6s} | Bar\n"
    header += "  " + "-" * 20 + "-+-" + "-" * 6 + "-+" + "-" * 20 + "\n"
    return header + "\n".join(rows) + "</pre>"
