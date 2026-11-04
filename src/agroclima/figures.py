"""Figuras estáticas del informe (PNG), generadas desde ``data/processed``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import config as C  # noqa: E402
from . import viz  # noqa: E402


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / f"{name}.parquet")


def _save(fig, name: str) -> str:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def _forest(ax, df: pd.DataFrame, label: str, est: str, lo: str, hi: str, color_by_sign: bool = True) -> None:
    df = df.reset_index(drop=True)
    y = np.arange(len(df))[::-1]
    for yi, (_, r) in zip(y, df.iterrows(), strict=True):
        excludes_zero = r[lo] > 0 or r[hi] < 0
        color = (viz.NINO if r[est] < 0 else viz.NINA) if color_by_sign and excludes_zero else viz.NEUTRAL
        ax.plot([r[lo], r[hi]], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.plot(r[est], yi, "o", color=color, ms=8, mec=viz.SURFACE, mew=2)
    ax.axvline(0, color=viz.AXIS, lw=1)
    ax.set_yticks(y, df[label])
    ax.grid(axis="y", visible=False)


def fig_anomalia_anual() -> str:
    yr = _load("anomalia_anual")
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.25}
    )
    colors = yr["fase"].map(viz.PHASE_COLORS).fillna(viz.NEUTRAL)
    axes[0].bar(yr["anio"], yr["oni"], color=colors, width=0.7)
    axes[0].axhline(0, color=viz.AXIS, lw=1)
    axes[0].set_ylabel("ONI medio (°C)")
    axes[0].set_title("Índice El Niño (ONI) por año")
    axes[1].bar(yr["anio"], yr["anom_pct"], color=colors, width=0.7)
    axes[1].axhline(0, color=viz.AXIS, lw=1)
    axes[1].set_ylabel("Anomalía de rendimiento (%)")
    axes[1].set_title("Rendimiento promedio frente a la tendencia de cada municipio-cultivo")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in viz.PHASE_COLORS.values()]
    fig.legend(
        handles,
        ["Año Niño", "Año Niña", "Año neutral"],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
    )
    axes[1].set_xticks(yr["anio"], [str(a) if a % 2 else "" for a in yr["anio"]])
    return _save(fig, "01_anomalia_anual")


def fig_efecto_cultivo() -> str:
    eff = _load("efectos_cultivo").sort_values("efecto_pct")
    eff = pd.concat([eff[eff["grupo"] == "Todos los cultivos"], eff[eff["grupo"] != "Todos los cultivos"]])
    fig, ax = plt.subplots(figsize=(8, 5.6))
    _forest(ax, eff, "grupo", "efecto_pct", "ic_low_pct", "ic_high_pct")
    ax.set_xlabel("Cambio en el rendimiento por cada +1 °C de ONI (%), IC 95 % bootstrap por años")
    ax.set_title("Efecto de El Niño sobre el rendimiento, por cultivo")
    ax.text(
        0,
        -0.13,
        "Rojo: pérdida con IC que excluye 0 · Azul: ganancia · Gris: no concluyente",
        transform=ax.transAxes,
        color=viz.MUTED,
        fontsize=9,
    )
    return _save(fig, "02_efecto_cultivo")


def fig_errores_estandar() -> str:
    nat = _load("efectos_cultivo").iloc[0]
    labels = [
        "MCO ingenuo\n(filas independientes)",
        "Agrupado por año\n(19 conglomerados)",
        "Bootstrap por años\n(usado)",
    ]
    vals = [nat["se_ingenuo"], nat["se_cluster_anio"], nat["se_bootstrap"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [viz.NEUTRAL, viz.NEUTRAL, viz.SERIES[0]]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f"  {v:.4f}", va="center", color=viz.INK_2)
    ax.set_xlabel("Error estándar del efecto nacional (log-rendimiento por °C)")
    ax.set_title("Por qué no basta el error estándar por defecto")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.3)
    return _save(fig, "03_errores_estandar")


def fig_mapa_lluvia() -> str:
    rain = _load("lluvia_vs_oni").merge(_load("municipios"), on="cod_mun")
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    cmap = LinearSegmentedColormap.from_list("div", viz.DIVERGING)
    lim = np.nanpercentile(np.abs(rain["lluvia_pct_por_oni"]), 98)
    sc = ax.scatter(
        rain["lon"],
        rain["lat"],
        c=rain["lluvia_pct_por_oni"],
        cmap=cmap,
        norm=TwoSlopeNorm(0, -lim, lim),
        s=14,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Cambio en la lluvia anual por +1 °C de ONI (%)", color=viz.INK_2)
    cb.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(
        "El Niño seca al país de forma desigual\n(lluvia CHIRPS 1991-2025, cada punto es un municipio)"
    )
    return _save(fig, "04_mapa_lluvia_oni")


def fig_region() -> str:
    het = _load("efectos_region")
    piv = het.pivot_table(index="cultivo", columns="region", values="efecto_pct_parcial")
    regions = [r.nombre for r in C.REGIONES if r.nombre in piv.columns]
    piv = piv[regions].loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 5.8))
    cmap = LinearSegmentedColormap.from_list("div", viz.DIVERGING)
    lim = np.nanmax(np.abs(piv.to_numpy()))
    im = ax.imshow(piv.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(regions)), regions)
    ax.set_yticks(range(len(piv)), piv.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=viz.INK if abs(v) < 0.6 * lim else "white",
            )
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("% de rendimiento por +1 °C de ONI", color=viz.INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Efecto por cultivo y región (agrupamiento parcial)")
    return _save(fig, "05_efecto_region")


def fig_canal() -> str:
    ch = _load("canal_climatico")
    ch = ch[ch["grupo"] == "Todos los cultivos"]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    _forest(ax, ch, "variable", "efecto_pct", "ic_low_pct", "ic_high_pct")
    ax.set_xlabel("Cambio en el rendimiento (%), IC 95 % agrupado por departamento")
    ax.set_title("Canal climático: comparando municipios dentro del mismo año")
    return _save(fig, "06_canal_climatico")


def fig_backtest() -> str:
    m = _load("backtest_metricas")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for name, color in viz.MODEL_COLORS.items():
        s = m[m["modelo"] == name]
        axes[0].plot(s["anio"], s["roc_auc"], "-o", color=color, ms=5, label=name)
        axes[1].plot(s["anio"], s["pr_auc"] / s["tasa_base"], "-o", color=color, ms=5, label=name)
    axes[0].axhline(0.5, color=viz.AXIS, lw=1)
    axes[0].set_title("ROC-AUC por año evaluado")
    axes[1].axhline(1, color=viz.AXIS, lw=1)
    axes[1].set_title("PR-AUC / tasa base (veces mejor que el azar)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    return _save(fig, "07_backtest")


def fig_calibracion() -> str:
    cal = _load("calibracion")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lim = max(cal["prob_media"].max(), cal["frecuencia_observada"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=viz.AXIS, lw=1)
    for version, color in (("Cruda", viz.NEUTRAL), ("Calibrada", viz.SERIES[0])):
        cv = cal[cal["version"] == version]
        ax.plot(
            cv["prob_media"],
            cv["frecuencia_observada"],
            "-o",
            color=color,
            ms=7,
            mec=viz.SURFACE,
            mew=2,
            label=version,
        )
    ax.legend(loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Probabilidad pronosticada")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración (deciles, backtest)")
    return _save(fig, "08_calibracion")


NICE = {
    "variante": "Cultivo (variante)",
    "region": "Región",
    "oni_jas_prev": "ONI jul-sep año previo",
    "oni_tend_prev": "Tendencia del ONI",
    "lluvia_anom_pct_prev": "Anomalía de lluvia ene-sep",
    "temp_anom_prev": "Anomalía de temperatura",
    "tmax_anom_prev": "Anomalía de temp. máxima",
    "meses_secos_prev": "Meses secos ene-sep",
    "meses_humedos_prev": "Meses húmedos ene-sep",
    "desvio_ultimo": "Último rendimiento vs. base",
    "volatilidad": "Volatilidad histórica",
    "tendencia": "Tendencia reciente",
    "tasa_malas_hist": "Frecuencia histórica de malas cosechas",
    "frac_repetidos": "Fracción de datos repetidos",
    "mala_ultima": "Última cosecha conocida fue mala",
    "anios_historia": "Años de historia",
    "log_area": "Área cosechada (log)",
    "mezcla_metodologia": "Base mezcla metodologías EVA",
    "lat": "Latitud",
    "lon": "Longitud",
    "elevacion_m": "Elevación",
}


def fig_shap() -> str:
    sh = _load("shap_global").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    is_enso = sh["variable"].str.startswith("oni")
    ax.barh(
        sh["variable"].map(NICE),
        sh["shap_medio_abs"],
        color=np.where(is_enso, viz.NINO, viz.SERIES[0]),
        height=0.6,
    )
    ax.set_xlabel("|SHAP| medio (log-odds)")
    ax.set_title("Qué mira el modelo (TreeSHAP, en rojo las variables de ENSO)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "09_shap_global")


def fig_riesgo() -> str:
    fc = _load("pronostico")
    mun = (
        fc.assign(ha=fc["prob_mala_cosecha"] * fc["area_ref"])
        .groupby("cod_mun")
        .agg(ha=("ha", "sum"), area=("area_ref", "sum"), lat=("lat", "first"), lon=("lon", "first"))
    )
    mun["prob"] = mun["ha"] / mun["area"]
    mun = mun.sort_values("prob")
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    cmap = LinearSegmentedColormap.from_list("risk", viz.SEQUENTIAL_RISK)
    sc = ax.scatter(
        mun["lon"],
        mun["lat"],
        c=mun["prob"] * 100,
        cmap=cmap,
        s=8 + 60 * np.sqrt(mun["area"] / mun["area"].max()),
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Probabilidad de mala cosecha (%, ponderada por área)", color=viz.INK_2)
    cb.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Riesgo de mala cosecha {C.FORECAST_YEAR}\n(tamaño = área cosechada)")
    return _save(fig, "10_riesgo_pronostico")


def build_all() -> list[str]:
    viz.mpl_style()
    return [
        fig_anomalia_anual(),
        fig_efecto_cultivo(),
        fig_errores_estandar(),
        fig_mapa_lluvia(),
        fig_region(),
        fig_canal(),
        fig_backtest(),
        fig_calibracion(),
        fig_shap(),
        fig_riesgo(),
    ]
