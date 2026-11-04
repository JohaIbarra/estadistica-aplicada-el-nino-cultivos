import streamlit as st

from charts import phase_bars
from data import load, model_meta, summary
from fmt import ci, num, pct, share

S = summary()
eff = load("efectos_cultivo")
het = load("efectos_region")
yr = load("anomalia_anual")
meta = model_meta()

nat = eff.iloc[0]
canal = load("canal_climatico")
hot = canal[(canal["variable"] == "Cada +1 °C de temperatura") & (canal["grupo"] != "Todos los cultivos")]
hurt = hot[hot["ic_high_pct"] < 0].sort_values("efecto_pct")
helped = hot[hot["ic_low_pct"] > 0]
sig = het[(het["ic_low_pct"] > 0) | (het["ic_high_pct"] < 0)]
losers = sig[sig["efecto_pct_parcial"] < 0]
winners = sig[sig["efecto_pct_parcial"] > 0]


def _names(df):
    # Cultivo, o cultivo (regiones) si el efecto no aplica a todas sus regiones
    out = []
    for crop, g in df.groupby("cultivo"):
        n_all = (het["cultivo"] == crop).sum()
        out.append(crop if len(g) == n_all else f"{crop} ({', '.join(g['region'])})")
    return ", ".join(out)


def _crop_effects(df):
    return ", ".join(f"{r.grupo.lower()} ({pct(r.efecto_pct)})" for r in df.itertuples())


def _who(row) -> str:
    n_all = (het["cultivo"] == row["cultivo"]).sum()
    same = het[(het["cultivo"] == row["cultivo"])]["efecto_pct_parcial"].round(2).nunique() == 1
    return row["cultivo"] if (same or n_all == 1) else f"{row['cultivo']} ({row['region']})"


year = meta["anio_pronostico"]
P = S["panel"]

st.title("El Niño, La Niña y los cultivos de Colombia")
st.markdown(
    f"<p class='nota'>Análisis de {num(P['filas'])} registros municipio-cultivo-año "
    f"({P['anios'][0]}–{P['anios'][1]}, {num(P['municipios'])} municipios, 12 cultivos) "
    "y sistema de alerta temprana de malas cosechas.</p>",
    unsafe_allow_html=True,
)

st.subheader("Las respuestas cortas")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Efecto nacional",
    pct(nat["efecto_pct"]),
    help="Cambio promedio del rendimiento por cada +1 °C de ONI, todos los cultivos.",
)
c1.caption(f"por °C de ONI · IC 95 %: {ci(nat['ic_low_pct'], nat['ic_high_pct'])}")
best = het.loc[het["efecto_pct_parcial"].idxmax()]
worst = het.loc[het["efecto_pct_parcial"].idxmin()]
c2.metric("Más favorecido", pct(best["efecto_pct_parcial"]))
c2.caption(f"{_who(best)} · IC 95 %: {ci(best['ic_low_pct'], best['ic_high_pct'])}")
c3.metric("Más afectado", pct(worst["efecto_pct_parcial"]))
c3.caption(f"{_who(worst)} · IC 95 %: {ci(worst['ic_low_pct'], worst['ic_high_pct'])}")
ha = S["pronostico"]["hectareas_en_riesgo"]
ha0 = S["pronostico"]["hectareas_en_riesgo_neutral"]
c4.metric(f"Ha en riesgo {year}", f"{num(ha / 1000)} mil")
c4.caption(
    f"+{num((ha - ha0) / 1000)} mil por El Niño frente a un año neutral · "
    f"ONI jul-sep 2026: {pct(meta['oni_jas_prev'], 2).replace(' %', ' °C')}"
)

st.markdown(
    f"""
<div class='hallazgo'><b>1. El Niño no es una catástrofe homogénea.</b> Para el agregado de los 12 cultivos, el
efecto es prácticamente cero y está acotado con precisión (entre {pct(nat["ic_low_pct"])} y
{pct(nat["ic_high_pct"])} por °C). El promedio esconde efectos de signo opuesto que se compensan.</div>
<div class='hallazgo'><b>2. Hay ganadores y perdedores.</b> Con intervalos que excluyen el cero, gana con El Niño:
{_names(winners) or "ninguno"}; pierden: {_names(losers) or "ninguno"}.
El café de montaña se beneficia de años más secos y soleados, consistente con la cosecha récord de 2015–2016.</div>
<div class='hallazgo'><b>3. La temperatura pesa más que la lluvia.</b> Comparando municipios dentro del mismo año,
cada +1 °C reduce el rendimiento de {_crop_effects(hurt) or "ningún cultivo"} y aumenta el de
{_crop_effects(helped) or "ningún cultivo"}; los efectos de la lluvia son más pequeños.</div>
<div class='hallazgo'><b>4. Los datos tienen límites que importan.</b> El {share(P["frac_repetidos"])} de los rendimientos
municipales de la EVA es idéntico al del año anterior (dato arrastrado), lo que atenúa cualquier señal climática.
El café, medido por la Federación de Cafeteros, casi no tiene este problema y es justamente donde la señal es más clara.</div>
""",
    unsafe_allow_html=True,
)

st.plotly_chart(
    phase_bars(yr, "anom_pct", "Rendimiento promedio vs. tendencia, por año (todos los cultivos)", " %"),
    use_container_width=True,
)
st.caption(
    "Cada barra es el promedio de las anomalías de rendimiento (respecto a la tendencia propia de cada "
    "municipio-cultivo, con un salto permitido en 2019 por el cambio metodológico de la EVA). "
    f"Años Niño: {', '.join(map(str, yr.loc[yr.fase == 'Niño', 'anio']))}. "
    f"Años Niña: {', '.join(map(str, yr.loc[yr.fase == 'Niña', 'anio']))}."
)

st.subheader("¿Y ahora?")
st.markdown(
    f"""
El ONI promedio de julio a septiembre de 2026 es **{pct(meta["oni_jas_prev"], 2).replace(" %", " °C")}**: un
episodio **El Niño fuerte**, más intenso que cualquiera del periodo de entrenamiento. La página
**Alerta temprana {year}** muestra qué municipios y cultivos tienen mayor probabilidad de una caída de
rendimiento de 20 % o más, cuánto de ese riesgo se debe a El Niño y un simulador de escenarios.
"""
)
