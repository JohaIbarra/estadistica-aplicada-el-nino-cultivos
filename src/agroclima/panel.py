"""Construcción del panel analítico municipio x variante de cultivo x año."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def filter_panel(annual: pd.DataFrame, outlier_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = [{"paso": "municipio-cultivo-año agregados", "filas": len(annual)}]
    df = annual[~outlier_mask]
    log.append(
        {
            "paso": f"rendimiento atípico (> {C.OUTLIER_MAD:g} MAD de su serie)",
            "filas": int(outlier_mask.sum()),
        }
    )
    small = df["area_cosechada"] < C.MIN_AREA_HA
    log.append({"paso": f"área cosechada < {C.MIN_AREA_HA:g} ha", "filas": int(small.sum())})
    df = df[~small]
    n_years = df.groupby("serie")["anio"].transform("nunique")
    short = n_years < C.MIN_YEARS
    log.append({"paso": f"series con menos de {C.MIN_YEARS} años", "filas": int(short.sum())})
    df = df[~short].reset_index(drop=True)
    log.append({"paso": "panel final", "filas": len(df)})
    return df, pd.DataFrame(log)


def detrend_log_yield(df: pd.DataFrame, break_2019: bool = False) -> pd.Series:
    """Anomalía de log-rendimiento: residuo de una tendencia lineal propia de cada serie.

    Restar la tendencia de cada municipio-cultivo elimina diferencias
    permanentes (suelo, altitud, tecnología) y mejoras graduales, de modo que lo
    que queda es la variación año a año que puede atribuirse al clima.

    Con ``break_2019=True`` se permite además un salto de nivel en 2019, cuando
    cambia la metodología de la EVA (análisis de sensibilidad).
    """
    y = np.log(df["rendimiento"].to_numpy())
    t = df["anio"].to_numpy(dtype=float)
    g = df["serie"].to_numpy()
    if not break_2019:
        X = np.column_stack([np.ones_like(t), t])
    else:
        X = np.column_stack([np.ones_like(t), t, (t >= 2019).astype(float)])
    resid = np.full(len(df), np.nan)
    order = np.argsort(g, kind="stable")
    g_sorted = g[order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        idx = order[a:b]
        Xi = X[idx]
        if break_2019 and (Xi[:, 2].min() == Xi[:, 2].max()):
            Xi = Xi[:, :2]  # la serie no cruza 2019: no hay salto que estimar
        beta, *_ = np.linalg.lstsq(Xi, y[idx], rcond=None)
        resid[idx] = y[idx] - Xi @ beta
    return pd.Series(resid, index=df.index, name="anom_log_rend")


def build_panel(
    annual: pd.DataFrame,
    enso_annual: pd.DataFrame,
    clima_mun: pd.DataFrame,
    municipios: pd.DataFrame,
    elevacion: pd.DataFrame,
) -> pd.DataFrame:
    """Une rendimiento, ENSO, clima y geografía.

    La anomalía principal (``anom_log_rend``) permite un salto de nivel en 2019
    porque la EVA cambió de metodología ese año: sin él, el promedio nacional de
    anomalías salta ~13 puntos log justo en 2019, un artefacto que se confundiría
    con el clima de ese año.
    """
    df = annual.merge(enso_annual, on="anio", how="left")
    df = df.merge(clima_mun, on=["cod_mun", "anio"], how="left")
    df = df.merge(municipios[["cod_mun", "lat", "lon"]], on="cod_mun", how="left")
    df = df.merge(elevacion, on="cod_mun", how="left")
    df["anom_log_rend"] = detrend_log_yield(df, break_2019=True)
    df["anom_log_rend_sin_quiebre"] = detrend_log_yield(df, break_2019=False)
    df["anom_pct"] = 100 * (np.exp(df["anom_log_rend"]) - 1)
    df["repetido"] = flag_repeated(df)
    return df


def flag_repeated(df: pd.DataFrame) -> pd.Series:
    """True si el rendimiento es idéntico al del año inmediatamente anterior.

    La EVA es un consenso municipal, no una medición. Un rendimiento idéntico
    al del año anterior suele indicar que el dato se arrastró sin
    actualizarse, lo que diluye cualquier señal climática.
    """
    d = df.sort_values(["serie", "anio"])
    prev_y = d.groupby("serie")["rendimiento"].shift()
    prev_t = d.groupby("serie")["anio"].shift()
    rep = np.isclose(d["rendimiento"], prev_y, rtol=1e-3) & (d["anio"] == prev_t + 1)
    return rep.reindex(df.index)
