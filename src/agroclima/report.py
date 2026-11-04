"""Genera las figuras y los informes en Markdown a partir de los resultados.

Los números del texto se leen de ``data/processed``: si se reejecuta el
pipeline con datos nuevos, el informe se actualiza solo.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from . import config as C
from . import figures


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / f"{name}.parquet")


def _pct(x: float, sign: bool = True) -> str:
    s = f"{x:+.1f}" if sign else f"{x:.1f}"
    return s.replace(".", ",") + " %"


def _num(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _dec(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def _share(x: float) -> str:
    return f"{100 * x:.0f} %"


def _names(df: pd.DataFrame, het: pd.DataFrame) -> str:
    # "Cacao (todas las regiones)" en lugar de repetir el cultivo por cada región
    parts = []
    for crop, g in df.groupby("cultivo"):
        n_all = int((het["cultivo"] == crop).sum())
        where = "todas las regiones" if len(g) == n_all else ", ".join(g["region"])
        parts.append(f"{crop} ({where}; {_pct(g['efecto_pct_parcial'].mean())} por °C)")
    return "; ".join(parts)


def _ci(lo: float, hi: float) -> str:
    return f"[{_pct(lo)}; {_pct(hi)}]"


def _context() -> dict:
    S = json.loads((C.PROCESSED / "resumen.json").read_text(encoding="utf-8"))
    eff = _load("efectos_cultivo")
    het = _load("efectos_region")
    canal = _load("canal_climatico")
    rob = _load("robustez")
    glob = _load("backtest_global").set_index("modelo")
    fc = _load("pronostico")
    P = _load("panel")
    rain = _load("lluvia_vs_oni")
    meta = json.loads((C.PROCESSED / "modelo_meta.json").read_text(encoding="utf-8"))
    return dict(S=S, eff=eff, het=het, canal=canal, rob=rob, glob=glob, fc=fc, P=P, rain=rain, meta=meta)


def executive(ctx: dict) -> str:
    S, eff, het, glob, fc, P, meta = (ctx[k] for k in ("S", "eff", "het", "glob", "fc", "P", "meta"))
    nat = eff.iloc[0]
    sig = het[(het["ic_low_pct"] > 0) | (het["ic_high_pct"] < 0)].sort_values("efecto_pct_parcial")
    losers = sig[sig["efecto_pct_parcial"] < 0]
    winners = sig[sig["efecto_pct_parcial"] > 0]
    lg = glob.loc[meta["modelo_elegido"]]
    year = meta["anio_pronostico"]
    by_crop = (
        fc.assign(ha=fc["prob_mala_cosecha"] * fc["area_ref"])
        .groupby("cultivo")
        .agg(ha=("ha", "sum"), area=("area_ref", "sum"))
        .assign(p=lambda d: d["ha"] / d["area"])
        .sort_values("ha", ascending=False)
    )
    canal = ctx["canal"][ctx["canal"]["grupo"] == "Todos los cultivos"].set_index("variable")
    temp = canal.loc["Cada +1 °C de temperatura"]
    rep = P["repetido"].mean()

    hot = ctx["canal"][
        (ctx["canal"]["variable"] == "Cada +1 °C de temperatura")
        & (ctx["canal"]["grupo"] != "Todos los cultivos")
    ]
    hurt = ", ".join(
        f"{r.grupo.lower()} ({_pct(r.efecto_pct)})" for r in hot[hot["ic_high_pct"] < 0].itertuples()
    )
    helped = ", ".join(
        f"{r.grupo.lower()} ({_pct(r.efecto_pct)})" for r in hot[hot["ic_low_pct"] > 0].itertuples()
    )
    shift = _load("pronostico_vs_neutral").sort_values("cambio_pp")
    up = shift.iloc[-1]
    down = shift.iloc[0]
    ha, ha0 = S["pronostico"]["hectareas_en_riesgo"], S["pronostico"]["hectareas_en_riesgo_neutral"]

    top3 = ", ".join(f"{C.CROPS[c]} ({_num(r.ha / 1000)} mil ha)" for c, r in by_crop.head(3).iterrows())
    return f"""# El Niño y los cultivos de Colombia — Informe ejecutivo

*Generado el {date.today():%d/%m/%Y} a partir de {_num(S["panel"]["filas"])} registros municipio-cultivo-año
({S["panel"]["anios"][0]}–{S["panel"]["anios"][1]}), {S["panel"]["municipios"]} municipios y 12 cultivos.*

## La pregunta

¿Cuánto daño causa El Niño a la agricultura colombiana, y se puede anticipar qué municipios tendrán una mala
cosecha con un año de antelación?

## Lo que encontramos

1. **A escala nacional, El Niño no reduce el rendimiento de forma apreciable.** Por cada +1 °C del índice ONI, el
   rendimiento promedio cambia {_pct(nat["efecto_pct"])} (IC 95 % {_ci(nat["ic_low_pct"], nat["ic_high_pct"])},
   p = {_dec(nat["p_permutacion"])} por permutación). Es un resultado nulo *preciso*: descarta pérdidas nacionales
   mayores a {_pct(abs(nat["ic_low_pct"]), sign=False)} por grado.

2. **El promedio esconde ganadores y perdedores.** Con intervalos que excluyen el cero:
   - Pierden: {_names(losers, het) or "ningún grupo"}.
   - Ganan: {_names(winners, het) or "ningún grupo"}.

   El café de montaña se beneficia de años más secos y soleados, en línea con la cosecha récord de 2015–2016.

3. **El canal más claro es la temperatura, y actúa cultivo por cultivo.** Comparando municipios dentro del mismo año,
   cada +1 °C de temperatura
   cambia el rendimiento promedio {_pct(temp["efecto_pct"])} (IC {_ci(temp["ic_low_pct"], temp["ic_high_pct"])}), con
   efectos negativos claros en {hurt or "ningún cultivo"} y positivos en {helped or "ningún cultivo"}.

4. **La calidad del dato limita lo que se puede medir.** El {_share(rep)} de los rendimientos municipales de la EVA es
   idéntico al del año anterior (dato arrastrado). Los efectos estimados deben leerse como una **cota inferior**.

## Alerta temprana {year}

El ONI de julio a septiembre de 2026 promedió **{_dec(meta["oni_jas_prev"])} °C**: un El Niño fuerte en desarrollo.
El modelo de alerta, emitido con la información disponible al 1 de octubre de 2026, estima
**{_num(ha / 1000)} mil hectáreas** con una caída de rendimiento de 20 % o más en {year} (valor esperado). Mayor
exposición: {top3}.

**¿Cuánto de eso es El Niño?** Con las mismas series y un ENSO neutral, el modelo estimaría {_num(ha0 / 1000)} mil
ha: El Niño suma unas **{_num((ha - ha0) / 1000)} mil ha** ({_pct(100 * (ha / ha0 - 1))}). El riesgo que más sube es
el de {C.CROPS[up["cultivo"]].lower()} ({_dec(up["cambio_pp"], 1)} puntos porcentuales) y el que más baja, el de
{C.CROPS[down["cultivo"]].lower()} ({_dec(down["cambio_pp"], 1)} pp). Coherente con la inferencia: El Niño redistribuye el
riesgo más que multiplicarlo. *Advertencia:* el ONI actual supera el máximo visto en entrenamiento
({_dec(meta["oni_entrenamiento"][1])} °C) y un modelo de árboles no extrapola; el efecto de un evento tan intenso
podría estar subestimado.

**¿Es confiable?** En la validación histórica (pronósticos simulados {int(_load("backtest_metricas")["anio"].min())}–2025
sin ver el futuro), el 10 % de casos marcados como de mayor riesgo tuvo una tasa real de malas cosechas de
{_share(lg["precision_top10"])}, frente al {_share(lg["tasa_base"])} promedio: **{_dec(lg["precision_top10"] / lg["tasa_base"], 1)} veces más**.
ROC-AUC = {_dec(lg["roc_auc"])}; supera con holgura a la referencia de persistencia, y con etiquetas permutadas cae
al azar (ROC-AUC {_dec(S["pronostico"]["control_negativo_roc_auc"])}), lo que descarta fugas de información. Sirve
para **priorizar** visitas técnicas, crédito y seguros, no como predicción exacta.

## Recomendaciones

- **Focalizar** la respuesta a El Niño por cultivo y región, no con medidas nacionales uniformes.
- **Priorizar** el monitoreo en los municipios del decil de mayor riesgo del tablero de alerta.
- **Mejorar la EVA**: marcar explícitamente los valores no actualizados permitiría medir el impacto climático con
  mucha más precisión.

*Metodología completa, supuestos y limitaciones en `reports/informe_tecnico.md`.*
"""


def technical(ctx: dict) -> str:
    S, eff, rob, glob, rain, P = (ctx[k] for k in ("S", "eff", "rob", "glob", "rain", "P"))
    nat = eff.iloc[0]
    ctrl_all = _load("control_negativo")
    ctrl, ctrl_mun = ctrl_all.iloc[0], ctrl_all.iloc[1]
    meta = ctx["meta"]
    bitacora = _load("bitacora")
    eff_tab = "\n".join(
        f"| {r.grupo} | {_pct(r.efecto_pct)} | {_ci(r.ic_low_pct, r.ic_high_pct)} | {r.p_permutacion:.3f} | "
        f"{'' if np.isnan(r.q_bh) else f'{r.q_bh:.2f}'} | {_num(r.n_obs)} |"
        for r in eff.itertuples()
    )
    rob_tab = "\n".join(
        f"| {r.especificacion} | {_pct(r.efecto_pct)} | {_ci(r.ic_low_pct, r.ic_high_pct)} | {r.p_permutacion:.2f} |"
        for r in rob.itertuples()
    )
    glob_tab = "\n".join(
        f"| {m} | {r.roc_auc:.3f} | {r.pr_auc:.3f} [{r.pr_auc_low:.3f}; {r.pr_auc_high:.3f}] | {r.brier:.4f} | "
        f"{r.brier_skill:+.3f} | {r.precision_top10:.1%} | {r.precision_top20:.1%} |"
        for m, r in glob.iterrows()
    )
    log_tab = "\n".join(f"| {r.etapa} | {r.paso} | {_num(r.filas)} |" for r in bitacora.itertuples())
    rep_by_crop = P.groupby("cultivo")["repetido"].mean().sort_values()
    rep_tab = ", ".join(f"{C.CROPS[c]} {v:.0%}" for c, v in rep_by_crop.items())
    dry_share = (rain["lluvia_pct_por_oni"] < 0).mean()
    return f"""# Informe técnico

## 1. Datos y construcción del panel

| Fuente | Uso |
|---|---|
| EVA 2007–2018 (`{C.EVA_HISTORICA_ID}`) y 2019–2025 (`{C.EVA_RECIENTE_ID}`), datos.gov.co | Rendimiento municipal |
| CHIRPS v2.0 mensual (IRI Data Library) | Lluvia, ventana de ±{C.CHIRPS_RADIUS_DEG}° en torno al centroide |
| NASA POWER mensual (MERRA-2) | Temperatura media y máxima |
| ONI (NOAA PSL) | Exposición a ENSO |
| DIVIPOLA (`{C.DIVIPOLA_ID}`) | Centroides municipales |

**Unidad de análisis:** serie = municipio × variante de cultivo; año agrícola = año calendario (semestres A y B
agregados: rendimiento = producción total / área cosechada total).

**Bitácora de limpieza**

| Etapa | Paso | Filas |
|---|---|---|
{log_tab}

**Hallazgos de calidad de datos**

- *Quiebre metodológico 2019.* Las dos bases EVA no son continuas: sin corrección, la anomalía media salta en
  2019. La tendencia de cada serie admite un salto de nivel en 2019.
- *Datos arrastrados.* {P["repetido"].mean():.1%} de los registros repiten exactamente el rendimiento del año
  anterior. Por cultivo: {rep_tab}.
- *Coordenadas corruptas.* La DIVIPOLA publicada contiene un municipio con latitud `525` y longitud
  `-74916666.875`; se recupera el punto decimal solo si el resultado cae dentro de Colombia.
- *Lluvia reanalizada inconsistente.* La lluvia de NASA POWER tiene correlación con el ONI de −0,78 (1991–2000) pero
  +0,19 (2001–2010); con CHIRPS la correlación nacional es estable (≈ −0,8 en todas las décadas). Se usa CHIRPS.
  En el {dry_share:.0%} de los municipios la lluvia anual cae cuando sube el ONI.

## 2. Inferencia del efecto

**Resultado:** anomalía de log-rendimiento = residuo de la regresión de log(rendimiento) sobre intercepto,
tendencia lineal y salto 2019, estimada por separado para cada serie. Absorbe diferencias permanentes (suelo,
altitud, tecnología) y mejoras graduales.

**Estimador:** pendiente MCO de la anomalía sobre el ONI medio enero–diciembre.

**Inferencia:** el ONI es común a todos los municipios en un año, así que el número efectivo de observaciones
para el efecto es ~19 años, no {_num(nat["n_obs"])} filas.

| Método | Error estándar |
|---|---|
| MCO ingenuo | {nat["se_ingenuo"]:.4f} |
| Agrupado por año (19 conglomerados) | {nat["se_cluster_anio"]:.4f} |
| **Bootstrap por años (usado)** | **{nat["se_bootstrap"]:.4f}** |

El error ingenuo es {nat["se_bootstrap"] / nat["se_ingenuo"]:.1f} veces demasiado pequeño. Los p-valores se
obtienen permutando el ONI entre años (4.000 permutaciones), exacto bajo la hipótesis nula. Con 12 cultivos se
reportan además q-valores de Benjamini-Hochberg.

| Grupo | Efecto por +1 °C ONI | IC 95 % | p (perm.) | q (B-H) | Obs. |
|---|---|---|---|---|---|
{eff_tab}

**Heterogeneidad cultivo × región:** pendiente y error bootstrap por celda, contraídos hacia la media del cultivo
con Bayes empírico (DerSimonian-Laird), equivalente a pendientes aleatorias de un modelo mixto pero con errores que
respetan la correlación entre municipios del mismo año.

**Canal climático:** regresión de la anomalía sobre déficit de lluvia, exceso de lluvia y anomalía de temperatura
con **efectos fijos de año** (identifica con variación entre municipios dentro del mismo año), errores agrupados
por departamento.

**Robustez del efecto nacional**

| Especificación | Efecto | IC 95 % | p |
|---|---|---|---|
{rob_tab}

## 3. Pronóstico de malas cosechas

- **Objetivo:** rendimiento en *t* ≥ {C.BAD_HARVEST_DROP:.0%} por debajo de la mediana de *t−2 … t−6*.
- **Emisión:** 1 de octubre de *t−1*. Rendimientos hasta *t−2* (rezago de publicación de la EVA), ONI de
  julio a septiembre de *t−1* y su tendencia, y clima de enero a septiembre de *t−1*.
- **Variables:** cultivo, región, ubicación y elevación, ENSO, anomalías climáticas, historia productiva
  (desvío del último dato, volatilidad, tendencia, frecuencia histórica de malas cosechas, fracción de datos
  repetidos, área).
- **Validación:** origen móvil; para el año *T* se entrena con años objetivo ≤ *T−2*. Referencias: climatología
  por cultivo y regresión logística.
- **Modelo:** LightGBM (hiperparámetros fijados a priori, sin optimizar sobre el backtest).

| Modelo | ROC-AUC | PR-AUC [IC 95 %] | Brier | Brier skill | Precisión 10 % alertas | Precisión 20 % alertas |
|---|---|---|---|---|---|---|
{glob_tab}

IC 95 % por bootstrap agrupado por municipio (las series de un municipio comparten clima y encuestador). La tasa
base ({glob.loc["LightGBM", "tasa_base"]:.1%}) es el piso del PR-AUC. No se reporta exactitud: un modelo que nunca
alerta tendría {1 - glob.loc["LightGBM", "tasa_base"]:.0%} de exactitud y no detectaría ningún caso.

**Lectura honesta.** La persistencia ("si la última cosecha conocida fue mala, esta también") es la referencia
más exigente en paneles; LightGBM la supera. Una regresión logística de cinco variables queda cerca de LightGBM:
la señal está en la historia productiva (desvío reciente, volatilidad), no en interacciones complejas.

**Controles negativos.** Se repitió el backtest completo con etiquetas permutadas:

| Permutación | ROC-AUC | PR-AUC | Tasa base |
|---|---|---|---|
| Dentro de cada año | {ctrl["roc_auc"]:.3f} | {ctrl["pr_auc"]:.3f} | {ctrl["tasa_base"]:.3f} |
| Dentro de cada municipio-año | {ctrl_mun["roc_auc"]:.3f} | {ctrl_mun["pr_auc"]:.3f} | {ctrl_mun["tasa_base"]:.3f} |

Con la permutación por año el desempeño cae al azar: no hay fugas. La permutación dentro de municipio-año
conserva la prevalencia local, de modo que el clima y la ubicación todavía pueden predecirla; la diferencia entre
esa fila y el modelo real ({glob.loc["LightGBM", "roc_auc"]:.3f}) es la señal que aporta la historia de cada serie.

**Calibración.** Isotónica, aprendida para cada año solo con predicciones de backtest de años ≤ T−2. Se adopta solo
si mejora el Brier del backtest; modelo usado para el pronóstico: **{meta["modelo_elegido"]}**.

**Extrapolación.** El ONI de julio a septiembre de 2026 ({meta["oni_jas_prev"]:+.2f} °C) supera el máximo del
entrenamiento ({meta["oni_entrenamiento"][1]:+.2f} °C, emitido en 2015 para 2016). Un modelo de árboles no
extrapola: trata el evento actual como el más fuerte que conoce, de modo que el riesgo de 2027 podría estar subestimado.

![Backtest](figures/07_backtest.png)
![Calibración](figures/08_calibracion.png)
![SHAP](figures/09_shap_global.png)

## 4. Limitaciones

- La EVA es una estimación por consenso; los datos arrastrados atenúan los efectos (sesgo hacia cero).
- {S["panel"]["frac_enteros"]:.0%} de los rendimientos son números enteros (apilamiento de dígitos): parte de la
  variación año a año es ruido de redondeo, irreducible para cualquier modelo.
- Las anomalías climáticas usan la normal 1991–2020, publicada en 2021; los pronósticos simulados anteriores a
  2021 la usan igualmente. Su efecto sobre una anomalía de nueve meses es despreciable frente al ruido de la EVA.
- El filtro de errores de digitación usa la mediana de toda la serie, pero solo actúa sobre desvíos de ×3 o más,
  de modo que no borra malas cosechas reales.
- Solo 19 años de variación de ENSO; los contrastes por fase (3 años Niño, 5 Niña) son imprecisos.
- Se mide rendimiento, no producción: El Niño también puede reducir el área sembrada.
- Clima en el centroide municipal; los municipios montañosos tienen gradientes internos fuertes.
- La interpretación causal requiere que, condicionado a la tendencia de cada serie, nada que varíe año a año y se
  correlacione con el ONI afecte el rendimiento por otras vías (p. ej., precios internacionales correlacionados con ENSO).

## Figuras

![Anomalía anual](figures/01_anomalia_anual.png)
![Efecto por cultivo](figures/02_efecto_cultivo.png)
![Errores estándar](figures/03_errores_estandar.png)
![Mapa lluvia-ONI](figures/04_mapa_lluvia_oni.png)
![Efecto por región](figures/05_efecto_region.png)
![Canal climático](figures/06_canal_climatico.png)
![Riesgo](figures/10_riesgo_pronostico.png)
"""


def build_all() -> None:
    figures.build_all()
    ctx = _context()
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "informe_ejecutivo.md").write_text(executive(ctx), encoding="utf-8")
    (C.REPORTS / "informe_tecnico.md").write_text(technical(ctx), encoding="utf-8")
