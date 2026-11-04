import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agroclima import config as C
from agroclima import viz
from agroclima.figures import NICE
from data import load, load_model, model_meta
from fmt import num, share

meta = model_meta()
year = meta["anio_pronostico"]
fc = load("pronostico").copy()
mun = load("municipios")[["cod_mun", "municipio", "departamento"]]
fc = fc.merge(mun, on="cod_mun", how="left")
fc["cultivo_nombre"] = fc["cultivo"].map(C.CROPS)
glob = load("backtest_global").set_index("modelo").loc[model_meta()["modelo_elegido"]]


@st.cache_resource(show_spinner=False)
def model():
    return load_model()


def predict(df: pd.DataFrame, oni: float, tend: float) -> np.ndarray:
    X = df[meta["features"]].copy()
    X["oni_jas_prev"] = oni
    X["oni_tend_prev"] = tend
    for c, cats in meta["categorias"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    p = model().predict(X)
    cal = meta["calibrador"]
    return np.interp(p, cal["x"], cal["y"]) if cal else p


st.title(f"Alerta temprana de malas cosechas · {year}")
st.markdown(
    f"""
<p class='nota'>Pronóstico emitido con la información disponible al 1 de octubre de 2026: ONI hasta septiembre,
lluvia y temperatura hasta septiembre, y rendimientos EVA hasta 2025. <b>Mala cosecha</b> = rendimiento al
menos {share(C.BAD_HARVEST_DROP)} por debajo de la mediana de los últimos 5 años publicados del mismo municipio-cultivo.
En la validación histórica, el 10 % de casos con mayor riesgo concentró una tasa de malas cosechas
{num(glob["precision_top10"] / glob["tasa_base"], 1)} veces la tasa base ({share(glob["precision_top10"])} frente a
{share(glob["tasa_base"])}).</p>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- escenario
with st.expander("Simulador de escenarios: ¿y si El Niño fuera más débil o más fuerte?", expanded=False):
    c1, c2 = st.columns(2)
    oni = c1.slider("ONI julio-septiembre 2026 (°C)", -2.0, 3.0, float(round(meta["oni_jas_prev"], 2)), 0.05)
    tend = c2.slider(
        "Tendencia del ONI (sep − jun, °C)", -1.5, 2.0, float(round(meta["oni_tend_prev"], 2)), 0.05
    )
    lo_tr, hi_tr = meta["oni_entrenamiento"]
    st.caption(
        f"Valores observados: ONI {meta['oni_jas_prev']:+.2f} °C, tendencia {meta['oni_tend_prev']:+.2f} °C. "
        "El simulador solo cambia las variables de ENSO; el resto (clima local, historia productiva) queda igual."
    )
    if oni > hi_tr or oni < lo_tr:
        st.warning(
            f"⚠ Fuera del rango visto en entrenamiento ({lo_tr:+.2f} a {hi_tr:+.2f} °C). Un modelo de árboles no "
            f"extrapola: trata cualquier ONI mayor a {hi_tr:+.2f} como el evento más fuerte que conoce (2015). "
            "El riesgo real de un Niño más intenso podría ser mayor que el mostrado."
        )
    scenario = (abs(oni - meta["oni_jas_prev"]) > 1e-9) or (abs(tend - meta["oni_tend_prev"]) > 1e-9)
    if scenario:
        fc["prob_mala_cosecha"] = predict(fc, oni, tend)
    grid = np.round(np.arange(-1.5, 3.01, 0.25), 2)
    curve = [(fc["area_ref"] * predict(fc, g, tend)).sum() / 1000 for g in grid]
    fig = go.Figure(
        go.Scatter(
            x=grid,
            y=curve,
            mode="lines+markers",
            line=dict(color=viz.SERIES[0], width=2),
            marker=dict(size=7),
            hovertemplate="ONI %{x:+.2f} °C<br>%{y:,.0f} mil ha en riesgo<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=hi_tr,
        x1=grid.max(),
        fillcolor=viz.MUTED,
        opacity=0.08,
        line_width=0,
        annotation_text="fuera del entrenamiento",
        annotation_position="bottom right",
        annotation_font_color=viz.MUTED,
    )
    fig.add_vline(
        x=meta["oni_jas_prev"],
        line=dict(color=viz.INK_2, width=1),
        annotation_text=f"observado ({meta['oni_jas_prev']:+.2f} °C)",
        annotation_position="top left",
    )
    fig.update_layout(
        **viz.plotly_layout(title="Hectáreas esperadas con mala cosecha según el ONI", height=280)
    )
    fig.update_xaxes(title="ONI julio-septiembre (°C)")
    fig.update_yaxes(title="miles de ha", rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- filtros
f1, f2, f3 = st.columns([2, 2, 1])
crops = sorted(fc["cultivo_nombre"].unique())
sel_crops = f1.multiselect("Cultivos", crops, default=crops)
deps = sorted(fc["departamento"].dropna().unique())
sel_deps = f2.multiselect("Departamentos", deps, placeholder="Todos")
min_ha = f3.number_input("Área mínima (ha)", 0, 10_000, 50, step=50)
view = fc[fc["cultivo_nombre"].isin(sel_crops) & (fc["area_ref"] >= min_ha)]
if sel_deps:
    view = view[view["departamento"].isin(sel_deps)]

expected_ha = (view["prob_mala_cosecha"] * view["area_ref"]).sum()
neutral_ha = (view["prob_neutral"] * view["area_ref"]).sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Series evaluadas",
    num(len(view)),
    help="Combinaciones municipio-cultivo con al menos 5 años de historia.",
)
k2.metric(
    "Ha en riesgo",
    num(expected_ha),
    help="Hectáreas esperadas con una caída de rendimiento de 20 % o más (suma de probabilidad × área).",
)
k3.metric(
    "Prob. media",
    share(expected_ha / max(view["area_ref"].sum(), 1)),
    help="Probabilidad de mala cosecha ponderada por área.",
)
k4.metric(
    "Por El Niño",
    ("+" if expected_ha >= neutral_ha else "−") + num(abs(expected_ha - neutral_ha)) + " ha",
    help="Diferencia frente al mismo pronóstico con ONI = 0 (ENSO neutral). Con el simulador abierto, "
    "compara el escenario elegido contra el neutral.",
)

with st.expander("¿Qué cultivos se vuelven más (o menos) riesgosos por El Niño?"):
    shift = (
        view.assign(
            ha=view["prob_mala_cosecha"] * view["area_ref"], ha0=view["prob_neutral"] * view["area_ref"]
        )
        .groupby("cultivo_nombre", as_index=False)
        .agg(area=("area_ref", "sum"), ha=("ha", "sum"), ha0=("ha0", "sum"))
    )
    shift["pp"] = 100 * (shift["ha"] - shift["ha0"]) / shift["area"]
    shift = shift.sort_values("pp")
    fig = go.Figure(
        go.Bar(
            x=shift["pp"],
            y=shift["cultivo_nombre"],
            orientation="h",
            marker=dict(color=np.where(shift["pp"] > 0, viz.NINO, viz.NINA), cornerradius=4),
            hovertemplate="%{y}: %{x:+.2f} puntos porcentuales<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=viz.AXIS, width=1))
    fig.update_layout(
        **viz.plotly_layout(
            height=380, title="Cambio en la probabilidad de mala cosecha frente a ENSO neutral"
        )
    )
    fig.update_xaxes(title="puntos porcentuales", ticksuffix=" pp")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Rojo: El Niño aumenta el riesgo; azul: lo reduce. El Niño redistribuye el riesgo entre cultivos más que "
        "multiplicarlo, coherente con el efecto nacional cercano a cero de la página de inferencia."
    )

# ---------------------------------------------------------------- mapa
m = (
    view.assign(ha=view["prob_mala_cosecha"] * view["area_ref"])
    .groupby(["cod_mun", "municipio", "departamento"], as_index=False)
    .agg(
        ha=("ha", "sum"),
        area=("area_ref", "sum"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        n=("serie", "size"),
    )
)
m["prob"] = 100 * m["ha"] / m["area"]
fig = px.scatter_map(
    m.sort_values("prob"),
    lat="lat",
    lon="lon",
    color="prob",
    size="area",
    size_max=22,
    color_continuous_scale=viz.SEQUENTIAL_RISK,
    range_color=(0, max(60, float(m["prob"].quantile(0.98)) if len(m) else 60)),
    hover_name="municipio",
    hover_data={
        "departamento": True,
        "prob": ":.0f",
        "area": ":,.0f",
        "ha": ":,.0f",
        "n": True,
        "lat": False,
        "lon": False,
    },
    labels={
        "prob": "Prob. mala cosecha (%)",
        "area": "Área (ha)",
        "ha": "Ha en riesgo",
        "n": "Cultivos",
        "departamento": "Depto.",
    },
    zoom=4.3,
    center={"lat": 5.5, "lon": -74.0},
    map_style="carto-positron",
    height=620,
)
fig.update_layout(**viz.plotly_layout(margin=dict(l=0, r=0, t=10, b=0)))
fig.update_coloraxes(colorbar=dict(title="%", thickness=12))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Color: probabilidad ponderada por área de los cultivos del municipio. Tamaño: área cosechada de referencia."
)

# ---------------------------------------------------------------- tablas
t1, t2 = st.columns([1, 1.6])
by_crop = (
    view.assign(ha=view["prob_mala_cosecha"] * view["area_ref"])
    .groupby("cultivo_nombre", as_index=False)
    .agg(ha=("ha", "sum"), area=("area_ref", "sum"))
    .assign(prob=lambda d: d["ha"] / d["area"])
    .sort_values("ha", ascending=False)
)
t1.markdown("**Por cultivo**")
t1.dataframe(
    by_crop.rename(
        columns={"cultivo_nombre": "Cultivo", "ha": "Ha en riesgo", "area": "Área (ha)", "prob": "Prob."}
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Ha en riesgo": st.column_config.NumberColumn(format="%.0f"),
        "Área (ha)": st.column_config.NumberColumn(format="%.0f"),
        "Prob.": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    },
)

top = view.sort_values("prob_mala_cosecha", ascending=False).head(200).copy()
top["factores"] = top["factores_principales"].map(
    lambda s: ", ".join(NICE.get(x.strip(), x) for x in s.split(","))
)
t2.markdown("**Casos de mayor riesgo** (factores = variables con mayor contribución SHAP)")
t2.dataframe(
    top[["municipio", "departamento", "variante", "prob_mala_cosecha", "area_ref", "factores"]].rename(
        columns={
            "municipio": "Municipio",
            "departamento": "Depto.",
            "variante": "Cultivo",
            "prob_mala_cosecha": "Prob.",
            "area_ref": "Área (ha)",
            "factores": "Factores principales",
        }
    ),
    hide_index=True,
    use_container_width=True,
    height=420,
    column_config={
        "Prob.": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Área (ha)": st.column_config.NumberColumn(format="%.0f"),
    },
)

csv = view[
    [
        "cod_mun",
        "municipio",
        "departamento",
        "cultivo_nombre",
        "variante",
        "area_ref",
        "prob_mala_cosecha",
        "factores_principales",
    ]
]
st.download_button(
    "Descargar pronóstico filtrado (CSV)",
    csv.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"pronostico_malas_cosechas_{year}.csv",
    mime="text/csv",
)
