"""Clima municipal: lluvia CHIRPS + temperatura NASA POWER, como anomalías.

Las anomalías (no los valores absolutos) son la señal relevante: 150 mm en un
mes es sequía en el Chocó y diluvio en La Guajira. Todas se calculan contra la
normal climatológica 1991-2020 del mismo mes y lugar.

Por qué dos fuentes: la lluvia de NASA POWER (reanálisis MERRA-2) no reproduce
la variabilidad interanual en Colombia (su correlación con El Niño cambia de
signo entre décadas), mientras que CHIRPS combina satélite con estaciones en
tierra y es la referencia en el trópico. La temperatura de MERRA-2 sí es
consistente y se conserva.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from . import config as C
from .download import snap_to_power_grid


# --------------------------------------------------------------------------
# Lectura de fuentes a tabla mensual por municipio
# --------------------------------------------------------------------------
def _read_chirps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as f:
        x = f.variables["X"][:].astype(float)
        y = f.variables["Y"][:].astype(float)
        t = f.variables["T"][:].astype(float)  # meses desde 1960-01
        p = f.variables["precipitation"][:].astype(float)
    p[(p < 0) | ~np.isfinite(p)] = np.nan  # océano / sin dato
    return x, y, t, p


def chirps_municipal(municipios: pd.DataFrame, cache_dir: Path | None = None) -> pd.DataFrame:
    """Lluvia mensual (mm) promediada en una ventana alrededor del centroide municipal."""
    cache_dir = cache_dir or C.RAW / "chirps"
    files = sorted(cache_dir.glob("chirps_*.nc"))
    if not files:
        raise FileNotFoundError("No hay archivos CHIRPS; ejecute `agroclima download`.")
    x, y, _, _ = _read_chirps(files[0])
    lat = municipios["lat"].to_numpy()
    lon = municipios["lon"].to_numpy()

    def window(radius: float) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (np.flatnonzero(np.abs(y - la) <= radius), np.flatnonzero(np.abs(x - lo) <= radius))
            for la, lo in zip(lat, lon, strict=True)
        ]

    small, large = window(C.CHIRPS_RADIUS_DEG), window(3 * C.CHIRPS_RADIUS_DEG)
    frames = []
    for f in files:
        _, _, t, p = _read_chirps(f)
        months = np.floor(t).astype(int)
        vals = np.full((len(municipios), len(t)), np.nan)
        for i, ((yi, xi), (yl, xl)) in enumerate(zip(small, large, strict=True)):
            block = p[:, yi][:, :, xi].reshape(len(t), -1)
            v = np.nanmean(block, axis=1) if np.isfinite(block).any() else np.full(len(t), np.nan)
            if np.isnan(v).all():  # municipios costeros/insulares: ventana más amplia
                block = p[:, yl][:, :, xl].reshape(len(t), -1)
                if np.isfinite(block).any():
                    v = np.nanmean(block, axis=1)
            vals[i] = v
        frame = pd.DataFrame(vals, columns=months)
        frame["cod_mun"] = municipios["cod_mun"].to_numpy()
        frames.append(frame.melt(id_vars="cod_mun", var_name="m", value_name="lluvia_mm"))
    out = pd.concat(frames, ignore_index=True)
    out["m"] = out["m"].astype(int)
    out["anio"] = 1960 + out["m"] // 12
    out["mes"] = out["m"] % 12 + 1
    return out.drop(columns="m").dropna(subset=["lluvia_mm"])


def power_municipal(power_long: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Temperatura media y máxima mensual (°C) de la celda POWER de cada municipio."""
    wide = power_long.pivot_table(
        index=["grid_lat", "grid_lon", "anio", "mes"], columns="variable", values="valor"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"T2M": "temp_c", "T2M_MAX": "tmax_c"})
    mun = municipios[["cod_mun", "lat", "lon"]].copy()
    mun["grid_lat"], mun["grid_lon"] = snap_to_power_grid(mun["lat"], mun["lon"])
    out = mun.merge(wide, on=["grid_lat", "grid_lon"], how="inner")
    return out[["cod_mun", "anio", "mes", "temp_c", "tmax_c"]]


def municipal_monthly(chirps: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    return chirps.merge(power, on=["cod_mun", "anio", "mes"], how="outer")


# --------------------------------------------------------------------------
# Anomalías y agregación anual
# --------------------------------------------------------------------------
def add_anomalies(monthly: pd.DataFrame, base: tuple[int, int] = C.CLIMATOLOGY) -> pd.DataFrame:
    """Anomalía de cada mes frente a la climatología del mismo mes y municipio."""
    ref = monthly[monthly["anio"].between(*base)]
    clim = (
        ref.groupby(["cod_mun", "mes"])
        .agg(
            lluvia_normal=("lluvia_mm", "mean"),
            temp_normal=("temp_c", "mean"),
            tmax_normal=("tmax_c", "mean"),
        )
        .reset_index()
    )
    out = monthly.merge(clim, on=["cod_mun", "mes"], how="left")
    out["temp_anom"] = out["temp_c"] - out["temp_normal"]
    out["tmax_anom"] = out["tmax_c"] - out["tmax_normal"]
    out["mes_seco"] = out["lluvia_mm"] < 0.5 * out["lluvia_normal"]
    out["mes_humedo"] = out["lluvia_mm"] > 1.5 * out["lluvia_normal"]
    return out


def _window(m: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    res = (
        m.groupby(keys)
        .agg(
            lluvia=("lluvia_mm", "sum"),
            lluvia_normal=("lluvia_normal", "sum"),
            temp_anom=("temp_anom", "mean"),
            tmax_anom=("tmax_anom", "mean"),
            meses_secos=("mes_seco", "sum"),
            meses_humedos=("mes_humedo", "sum"),
            n_lluvia=("lluvia_mm", "count"),
            n_temp=("temp_anom", "count"),
        )
        .reset_index()
    )
    res["lluvia_anom_pct"] = 100 * (res["lluvia"] / res["lluvia_normal"] - 1)
    return res


def annual_features(monthly_anom: pd.DataFrame) -> pd.DataFrame:
    """Variables climáticas anuales por municipio.

    Sufijo ``_anual`` = enero-diciembre del año t (inferencia del efecto).
    Sufijo ``_prev`` = enero-septiembre del año t-1, lo conocido el 1 de octubre
    de t-1, que es la fecha de emisión del pronóstico para t.
    """
    keys = ["cod_mun", "anio"]
    drop = ["n_lluvia", "n_temp", "lluvia", "lluvia_normal"]

    full = _window(monthly_anom, keys)
    full = full[(full["n_lluvia"] == 12) & (full["n_temp"] == 12)].drop(columns=drop)
    full = full.add_suffix("_anual").rename(columns={f"{k}_anual": k for k in keys})

    last = C.FORECAST_LAST_CLIMATE_MONTH
    prev = _window(monthly_anom[monthly_anom["mes"] <= last], keys)
    prev = prev[(prev["n_lluvia"] == last) & (prev["n_temp"] == last)].drop(columns=drop)
    prev["anio"] = prev["anio"] + 1
    prev = prev.add_suffix("_prev").rename(columns={f"{k}_prev": k for k in keys})

    return full.merge(prev, on=keys, how="outer")


def municipal_elevation(elev: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Elevación media de la celda POWER de cada municipio (aproximación gruesa)."""
    mun = municipios[["cod_mun", "lat", "lon"]].copy()
    mun["grid_lat"], mun["grid_lon"] = snap_to_power_grid(mun["lat"], mun["lon"])
    return mun.merge(elev, on=["grid_lat", "grid_lon"], how="left")[["cod_mun", "elevacion_m"]]
