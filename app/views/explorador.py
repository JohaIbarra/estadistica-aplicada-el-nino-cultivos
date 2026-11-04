import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agroclima import config as C
from agroclima import viz
from data import load
from fmt import pct, share

P = load("panel_app").copy()
P["cultivo_nombre"] = P["cultivo"].map(C.CROPS)
enso = load("enso_anual").set_index("anio")

st.title("Explorador de datos")
st.markdown(
    "<p class='nota'>Anomalía = rendimiento del año frente a la tendencia propia del municipio-cultivo. "
    "Rojo: por debajo de lo esperado; azul: por encima.</p>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 2])
crop = c1.selectbox(
    "Cultivo", sorted(P["cultivo_nombre"].unique()), index=sorted(P["cultivo_nombre"].unique()).index("Café")
)
years = sorted(P["anio"].unique())
year = c2.select_slider("Año", years, value=2015)
fase = enso.loc[year, "fase"]
st.markdown(
    f"**{year}**: ONI medio {pct(enso.loc[year, 'oni_anual'], 2).replace(' %', '')} °C · año **{fase}**"
)

d = P[(P["cultivo_nombre"] == crop) & (P["anio"] == year)].dropna(subset=["lat"])
d = d.groupby(["cod_mun", "municipio", "departamento"], as_index=False).agg(
    anom_pct=("anom_pct", "mean"),
    area=("area_cosechada", "sum"),
    rend=("rendimiento", "mean"),
    lat=("lat", "first"),
    lon=("lon", "first"),
    repetido=("repetido", "max"),
)
lim = 40.0
fig = px.scatter_map(
    d,
    lat="lat",
    lon="lon",
    color=d["anom_pct"].clip(-lim, lim),
    size="area",
    size_max=20,
    hover_name="municipio",
    hover_data={
        "departamento": True,
        "anom_pct": ":+.1f",
        "rend": ":.2f",
        "area": ":,.0f",
        "repetido": True,
        "lat": False,
        "lon": False,
    },
    labels={
        "color": "Anomalía %",
        "anom_pct": "Anomalía %",
        "rend": "Rend. t/ha",
        "area": "Área ha",
        "repetido": "Igual al año previo",
        "departamento": "Depto.",
    },
    color_continuous_scale=viz.DIVERGING,
    range_color=(-lim, lim),
    color_continuous_midpoint=0,
    zoom=4.3,
    center={"lat": 5.5, "lon": -74.0},
    map_style="carto-positron",
    height=560,
)
fig.update_layout(**viz.plotly_layout(margin=dict(l=0, r=0, t=10, b=0)))
st.plotly_chart(fig, use_container_width=True)

k1, k2, k3 = st.columns(3)
w = d["area"] / d["area"].sum()
k1.metric("Municipios con dato", len(d))
k2.metric("Anomalía media (ponderada por área)", f"{pct((w * d['anom_pct']).sum(), 1).replace(' %', '')} %")
k3.metric("Datos idénticos al año anterior", f"{share(d['repetido'].mean(), 0)}")

st.subheader("Serie de un municipio")
opts = (
    P[P["cultivo_nombre"] == crop].groupby(["cod_mun", "municipio", "departamento"])["area_cosechada"].mean()
)
opts = opts.sort_values(ascending=False).reset_index()
label = opts["municipio"] + " (" + opts["departamento"] + ")"
pick = st.selectbox("Municipio (ordenados por área)", label.tolist())
cod = opts.loc[label == pick, "cod_mun"].iloc[0]
s = P[(P["cod_mun"] == cod) & (P["cultivo_nombre"] == crop)].sort_values("anio")
fig = go.Figure()
for variante, g in s.groupby("variante"):
    fig.add_trace(
        go.Scatter(
            x=g["anio"],
            y=g["rendimiento"],
            mode="lines+markers",
            name=variante.title(),
            marker=dict(
                size=8,
                color=np.where(g["repetido"], viz.SURFACE, viz.SERIES[0]),
                line=dict(color=viz.SERIES[0], width=2),
            ),
            line=dict(color=viz.SERIES[0], width=2),
            hovertemplate="%{x}: %{y:.2f} t/ha<extra></extra>",
        )
    )
for y_ in enso.index[(enso.index >= years[0]) & (enso.index <= years[-1])]:
    if enso.loc[y_, "fase"] == "Niño":
        fig.add_vrect(x0=y_ - 0.5, x1=y_ + 0.5, fillcolor=viz.NINO, opacity=0.08, line_width=0)
    elif enso.loc[y_, "fase"] == "Niña":
        fig.add_vrect(x0=y_ - 0.5, x1=y_ + 0.5, fillcolor=viz.NINA, opacity=0.08, line_width=0)
fig.add_vline(
    x=2018.5,
    line=dict(color=viz.MUTED, width=1),
    annotation_text="cambio metodológico EVA",
    annotation_font_color=viz.MUTED,
)
fig.update_layout(**viz.plotly_layout(title=f"{crop} en {pick}", height=340))
fig.update_yaxes(title="t/ha", rangemode="tozero")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Franjas rojas: años Niño; azules: años Niña. Puntos huecos: rendimiento idéntico al año anterior (probable dato arrastrado)."
)
