import plotly.graph_objects as go
import streamlit as st

from agroclima import config as C
from agroclima import viz
from agroclima.figures import NICE
from data import load, model_meta
from fmt import num, share

bt = load("backtest_metricas")
glob = load("backtest_global")
cal = load("calibracion")
shap = load("shap_global")
log = load("bitacora")

st.title("Metodología y validación")

st.header("1. Datos")
st.markdown(
    """
| Fuente | Qué aporta | Resolución |
|---|---|---|
| **EVA** – Evaluaciones Agropecuarias Municipales (MinAgricultura, datos.gov.co) | Área, producción y rendimiento por municipio y cultivo, 2007-2025 | municipio · año/semestre |
| **CHIRPS v2.0** (UCSB) vía IRI Data Library | Lluvia mensual (satélite + estaciones) | 0,05° (~5 km) |
| **NASA POWER** (MERRA-2) | Temperatura media y máxima mensual | 0,5° × 0,625° |
| **ONI** (NOAA CPC/PSL) | Índice oficial de El Niño / La Niña | mensual |
| **DIVIPOLA** (MinSalud) | Centroide de cada municipio | municipio |
"""
)
with st.expander("Bitácora de limpieza: cuántos registros se descartaron y por qué"):
    st.dataframe(log, hide_index=True, use_container_width=True)
st.markdown(
    """
**Decisiones de limpieza que cambian resultados:**
- *Dos bases EVA con esquemas distintos* (2007-2018 y 2019-2025): se armonizan nombres, tildes, códigos DANE y
  variantes (p. ej., "maíz amarillo tradicional" + "maíz blanco tradicional" = "maíz tradicional").
- *Variantes separadas*: maíz tecnificado (~4 t/ha) y tradicional (~1,5 t/ha) no se mezclan, para que un cambio en
  la composición del área no parezca un cambio de rendimiento.
- *Quiebre metodológico en 2019*: el promedio de anomalías salta justo cuando cambia la base. La tendencia de cada
  serie permite un salto de nivel en 2019.
- *Coordenadas corruptas en la DIVIPOLA* (p. ej., latitud `525`): se recupera el punto decimal si el resultado cae
  dentro de Colombia.
- *Lluvia*: la de NASA POWER se descartó porque su correlación con El Niño cambia de signo entre décadas;
  CHIRPS mantiene una correlación nacional estable (≈ −0,8).
"""
)

st.header("2. Pronóstico: diseño sin fuga de información")
st.markdown(
    f"""
- **Fecha de emisión:** 1 de octubre del año anterior al pronosticado.
- **Rendimientos:** solo hasta *t−{C.EVA_PUBLICATION_LAG}*, porque la EVA de *t−1* aún no está publicada.
- **Clima y ONI:** solo hasta septiembre de *t−1*.
- **Validación con origen móvil:** para evaluar el año *T* se entrena solo con años objetivo ≤ *T−2*.
- Cada regla tiene un test automático (`tests/test_predict.py`) que altera datos "futuros" y verifica que las variables no cambian.
"""
)

g = glob.set_index("modelo")
ctrls = load("control_negativo")
ctrl, ctrl_mun = ctrls.iloc[0], ctrls.iloc[1]
tabla = glob[
    [
        "modelo",
        "roc_auc",
        "pr_auc",
        "pr_auc_low",
        "pr_auc_high",
        "brier_skill",
        "precision_top10",
        "precision_top20",
    ]
]
st.dataframe(
    tabla.rename(
        columns={
            "modelo": "Modelo",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "pr_auc_low": "PR-AUC IC inf.",
            "pr_auc_high": "PR-AUC IC sup.",
            "brier_skill": "Brier skill",
            "precision_top10": "Precisión con 10 % de alertas",
            "precision_top20": "Precisión con 20 % de alertas",
        }
    ),
    hide_index=True,
    use_container_width=True,
    column_config={
        c: st.column_config.NumberColumn(format="%.3f")
        for c in [
            "ROC-AUC",
            "PR-AUC",
            "PR-AUC IC inf.",
            "PR-AUC IC sup.",
            "Brier skill",
            "Precisión con 10 % de alertas",
            "Precisión con 20 % de alertas",
        ]
    },
)
st.caption(
    f"Backtest {int(bt['anio'].min())}-{int(bt['anio'].max())}. Tasa base de malas cosechas: "
    f"{share(g.loc['LightGBM', 'tasa_base'], 1)} (es el piso del PR-AUC). IC 95 % por bootstrap agrupado por municipio. "
    "No se reporta exactitud (accuracy): un modelo que nunca alerta acertaría el 85 % y no detectaría nada."
)
st.markdown(
    f"""
**Control negativo.** Se permutaron las etiquetas dentro de cada año y se repitió todo el backtest. Si hubiera
fuga de información, el modelo seguiría "acertando". Resultado: ROC-AUC = **{num(ctrl["roc_auc"], 3)}** y
PR-AUC = **{num(ctrl["pr_auc"], 3)}** frente a una tasa base de {num(ctrl["tasa_base"], 3)}: cae al azar, como debe.
Permutando solo dentro de cada municipio-año (se conserva la prevalencia local) queda ROC-AUC = {num(ctrl_mun["roc_auc"], 3)}:
esa es la parte de la señal que aportan el clima y la ubicación; el resto viene de la historia de cada serie.

**Persistencia como referencia.** "Si la última cosecha conocida fue mala, esta también" es la línea base que más
cuesta superar en paneles. LightGBM la supera en PR-AUC ({num(g.loc["LightGBM", "pr_auc"], 3)} frente a
{num(g.loc["Persistencia (referencia)", "pr_auc"], 3)}). Una regresión logística con 5 variables queda cerca
({num(g.loc["Regresión logística", "pr_auc"], 3)}): la señal está sobre todo en la historia productiva, no en interacciones complejas.
"""
)

c1, c2 = st.columns([1.6, 1])
fig = go.Figure()
for name, color in viz.MODEL_COLORS.items():
    s_ = bt[bt["modelo"] == name]
    fig.add_trace(
        go.Scatter(
            x=s_["anio"],
            y=s_["pr_auc"] / s_["tasa_base"],
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.2f}} veces la tasa base<extra></extra>",
        )
    )
fig.add_hline(y=1, line=dict(color=viz.AXIS, width=1))
fig.update_layout(**viz.plotly_layout(title="PR-AUC / tasa base, por año evaluado", height=360))
fig.update_yaxes(rangemode="tozero")
c1.plotly_chart(fig, use_container_width=True)

lim = float(max(cal["prob_media"].max(), cal["frecuencia_observada"].max()) * 1.1)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, lim],
        y=[0, lim],
        mode="lines",
        line=dict(color=viz.AXIS, width=1),
        hoverinfo="skip",
        showlegend=False,
    )
)
for version, color in (("Cruda", viz.NEUTRAL), ("Calibrada", viz.SERIES[0])):
    cv = cal[cal["version"] == version]
    fig.add_trace(
        go.Scatter(
            x=cv["prob_media"],
            y=cv["frecuencia_observada"],
            mode="lines+markers",
            name=version,
            line=dict(color=color, width=2),
            marker=dict(size=9, line=dict(color=viz.SURFACE, width=2)),
            hovertemplate=f"{version}<br>Pronosticado %{{x:.0%}}<br>Observado %{{y:.0%}}<extra></extra>",
        )
    )
fig.update_layout(**viz.plotly_layout(title="Calibración (deciles)", height=360))
fig.update_xaxes(title="Probabilidad pronosticada", tickformat=".0%", range=[0, lim])
fig.update_yaxes(title="Frecuencia observada", tickformat=".0%", range=[0, lim])
c2.plotly_chart(fig, use_container_width=True)
c2.caption(
    "Calibración isotónica aprendida solo con años ya evaluables al emitir cada pronóstico. "
    f"Se adopta solo si mejora el Brier del backtest; modelo usado: {model_meta()['modelo_elegido']}."
)

sh = shap.head(12).iloc[::-1]
fig = go.Figure(
    go.Bar(
        x=sh["shap_medio_abs"],
        y=sh["variable"].map(NICE),
        orientation="h",
        marker=dict(
            color=[viz.NINO if v.startswith("oni") else viz.SERIES[0] for v in sh["variable"]], cornerradius=4
        ),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    )
)
fig.update_layout(
    **viz.plotly_layout(title="Importancia de variables (|SHAP| medio; en rojo, ENSO)", height=420)
)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.header("3. Limitaciones")
st.markdown(
    """
- **La EVA es una estimación por consenso, no una medición.** Un tercio de los rendimientos son idénticos al año
  anterior. Esto atenúa los efectos climáticos estimados: son una cota inferior del efecto real.
- **El Niño 2026 es más intenso que cualquier caso de entrenamiento** (ONI jul-sep +2,13 frente a un máximo de
  +1,73 en 2015). El modelo de árboles no extrapola: el riesgo de 2027 podría estar subestimado.
- **Rendimientos redondeados.** Cerca del 40 % de los rendimientos son números enteros: parte de la variación año a
  año es ruido de redondeo, irreducible para cualquier modelo.
- **Solo 19 años de variación de ENSO** (3 años Niño y 5 Niña según el promedio anual). Los efectos por fase son imprecisos por construcción.
- **Rendimiento ≠ producción.** El Niño también puede reducir el área sembrada o cosechada; este estudio mide rendimiento por hectárea.
- **Clima en el centroide municipal.** Municipios grandes y montañosos tienen climas internos muy distintos.
- **El pronóstico identifica riesgo relativo**, útil para priorizar; las probabilidades absolutas dependen de la calibración histórica.
"""
)
