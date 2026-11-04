"""Sistema visual compartido por el informe (matplotlib) y el dashboard (plotly).

Reglas: un solo eje Y por gráfica, marcas delgadas, rejilla tenue, color
según la función del dato (identidad, magnitud o polaridad), nunca por rango.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categórico (orden fijo, validado para daltonismo en los 3 primeros puestos)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {
    "LightGBM": SERIES[0],
    "Regresión logística": SERIES[1],
    "Persistencia (referencia)": SERIES[2],
    "Climatología (referencia)": MUTED,
}

# Polaridad: El Niño (cálido) frente a La Niña (fría), neutral en gris.
NINO = "#e34948"
NINA = "#2a78d6"
NEUTRAL = "#898781"  # gris de énfasis: "nada que reportar", contraste 3.5:1
PHASE_COLORS = {"Niño": NINO, "Niña": NINA, "Neutral": NEUTRAL}

# Divergente rojo <-> azul con punto medio gris (pérdida <-> ganancia, seco <-> húmedo)
DIVERGING = ["#a32b2b", "#e34948", "#f0a3a0", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]
# Secuencial de una sola tonalidad (riesgo), de claro a oscuro
SEQUENTIAL_RISK = ["#fdeee6", "#f9c9b0", "#f29e78", "#eb6834", "#c44d1d", "#8f3511"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FONT = "Segoe UI, system-ui, -apple-system, sans-serif"


def mpl_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def plotly_layout(**over) -> dict:
    base = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=16), x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK_2)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK_2)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    base.update(over)
    return base
