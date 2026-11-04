"""Gráficas plotly reutilizables del dashboard (con tooltips por defecto)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from agroclima import viz
from data import ROOT  # noqa: F401  (asegura sys.path hacia src/)


def forest(
    df: pd.DataFrame, label: str, est: str, lo: str, hi: str, x_title: str, height: int | None = None
) -> go.Figure:
    df = df.reset_index(drop=True)
    excl = (df[lo] > 0) | (df[hi] < 0)
    colors = np.where(excl & (df[est] < 0), viz.NINO, np.where(excl, viz.NINA, viz.NEUTRAL))
    fig = go.Figure()
    for i, r in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[r[lo], r[hi]],
                y=[r[label]] * 2,
                mode="lines",
                line=dict(color=colors[i], width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df[est],
            y=df[label],
            mode="markers",
            marker=dict(color=colors, size=11, line=dict(color=viz.SURFACE, width=2)),
            customdata=np.stack([df[lo], df[hi]], axis=1),
            hovertemplate="<b>%{y}</b><br>Efecto: %{x:+.1f} %<br>IC 95 %: %{customdata[0]:+.1f} a %{customdata[1]:+.1f} %<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line=dict(color=viz.AXIS, width=1))
    fig.update_layout(
        **viz.plotly_layout(height=height or 60 + 34 * len(df)),
    )
    fig.update_xaxes(title=x_title, ticksuffix=" %")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def phase_bars(yr: pd.DataFrame, value: str, title: str, suffix: str) -> go.Figure:
    fig = go.Figure()
    for fase, color in viz.PHASE_COLORS.items():
        s = yr[yr["fase"] == fase]
        fig.add_trace(
            go.Bar(
                x=s["anio"],
                y=s[value],
                name=f"Año {fase.lower() if fase == 'Neutral' else fase}",
                marker=dict(color=color, cornerradius=4),
                hovertemplate=f"%{{x}} · {fase}<br>%{{y:+.2f}}{suffix}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color=viz.AXIS, width=1))
    fig.update_layout(**viz.plotly_layout(title=title, height=300, bargap=0.25))
    fig.update_yaxes(ticksuffix=suffix)
    return fig
