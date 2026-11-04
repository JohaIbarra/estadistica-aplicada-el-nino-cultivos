"""Descarga reproducible de las fuentes públicas, con caché en ``data/raw``.

Cada descarga es idempotente: si el archivo ya existe no se vuelve a pedir
(usar ``force=True`` para refrescar). Así el pipeline completo se puede
reejecutar sin castigar las APIs públicas.
"""

from __future__ import annotations

import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "agroclima-portfolio/1.0 (analisis academico)"}


def _get(url: str, *, params: dict | None = None, retries: int = 5, timeout: int = 180) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            log.warning("Fallo %s (%s); reintento en %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError("inalcanzable")


def _socrata_csv(dataset_id: str, dest: Path, force: bool = False, page: int = 100_000) -> Path:
    """Descarga un dataset Socrata completo paginando por ``:id`` (orden estable)."""
    if dest.exists() and not force:
        return dest
    frames, offset = [], 0
    while True:
        r = _get(
            f"{C.SOCRATA}/{dataset_id}.csv",
            params={"$limit": page, "$offset": offset, "$order": ":id"},
        )
        chunk = pd.read_csv(io.StringIO(r.text), dtype=str)
        frames.append(chunk)
        log.info("%s: %d filas (offset %d)", dataset_id, len(chunk), offset)
        if len(chunk) < page:
            break
        offset += page
    df = pd.concat(frames, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def download_eva(force: bool = False) -> tuple[Path, Path]:
    hist = _socrata_csv(C.EVA_HISTORICA_ID, C.RAW / "eva_2007_2018.csv", force)
    rec = _socrata_csv(C.EVA_RECIENTE_ID, C.RAW / "eva_2019_2025.csv", force)
    return hist, rec


def download_divipola(force: bool = False) -> Path:
    return _socrata_csv(C.DIVIPOLA_ID, C.RAW / "divipola_municipios.csv", force)


def download_oni(force: bool = False) -> Path:
    dest = C.RAW / "oni.data"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_get(C.ONI_URL).text, encoding="utf-8")
    return dest


def snap_to_power_grid(lat: pd.Series, lon: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Centro de la celda MERRA-2 que contiene cada coordenada."""
    glat = (np.round(lat / C.POWER_GRID_LAT) * C.POWER_GRID_LAT).round(3)
    glon = (np.round(lon / C.POWER_GRID_LON) * C.POWER_GRID_LON).round(3)
    return glat, glon


def _power_cell(lat: float, lon: float, cache_dir: Path) -> Path:
    dest = cache_dir / f"{lat:+.3f}_{lon:+.3f}.json"
    if dest.exists():
        return dest
    r = _get(
        C.POWER_URL,
        params={
            "parameters": ",".join(C.POWER_PARAMS),
            "community": "AG",
            "latitude": lat,
            "longitude": lon,
            "start": C.POWER_START,
            "end": C.POWER_END,
            "format": "JSON",
        },
    )
    dest.write_text(r.text, encoding="utf-8")
    return dest


def power_cache_dir() -> Path:
    # El rango de años va en el nombre: cambiarlo invalida la caché automáticamente.
    return C.RAW / f"power_{C.POWER_START}_{C.POWER_END}"


def download_power(municipios: pd.DataFrame, workers: int = 6) -> Path:
    """Clima mensual para cada celda de la malla que contiene al menos un municipio."""
    cache = power_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    glat, glon = snap_to_power_grid(municipios["lat"], municipios["lon"])
    cells = sorted(set(zip(glat, glon, strict=True)))
    log.info("NASA POWER: %d celdas únicas para %d municipios", len(cells), len(municipios))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_power_cell, la, lo, cache): (la, lo) for la, lo in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 50 == 0:
                log.info("NASA POWER: %d/%d celdas", i, len(cells))
    return cache


def load_power_cache(cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee la caché JSON de POWER.

    Devuelve (clima en formato largo celda x mes x variable, elevación por celda).
    """
    cache_dir = cache_dir or power_cache_dir()
    rows, elev = [], []
    for f in sorted(cache_dir.glob("*.json")):
        lat_s, lon_s = f.stem.split("_")
        lat, lon = float(lat_s), float(lon_s)
        payload = json.loads(f.read_text(encoding="utf-8"))
        elev.append((lat, lon, payload["geometry"]["coordinates"][2]))
        for var, series in payload["properties"]["parameter"].items():
            for key, val in series.items():
                month = int(key[4:])
                if month == 13:  # POWER añade el promedio anual como mes "13"
                    continue
                rows.append((lat, lon, int(key[:4]), month, var, val))
    df = pd.DataFrame(rows, columns=["grid_lat", "grid_lon", "anio", "mes", "variable", "valor"])
    df.loc[df["valor"] <= -999, "valor"] = np.nan  # valor de relleno de POWER
    elev_df = pd.DataFrame(elev, columns=["grid_lat", "grid_lon", "elevacion_m"])
    return df.dropna(subset=["valor"]), elev_df


def download_chirps(force: bool = False) -> Path:
    """Lluvia CHIRPS mensual sobre Colombia, un netCDF por año (reintentable)."""
    cache = C.RAW / "chirps"
    cache.mkdir(parents=True, exist_ok=True)
    lon0, lon1, lat0, lat1 = C.CHIRPS_BOX
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for year in range(C.POWER_START, C.POWER_END + 1):
        dest = cache / f"chirps_{year}.nc"
        if dest.exists() and not force:
            continue
        last = months[C.FORECAST_LAST_CLIMATE_MONTH - 1] if year == C.POWER_END else "Dec"
        url = (
            f"{C.CHIRPS_URL}/X/{lon0}/{lon1}/RANGEEDGES/Y/{lat0}/{lat1}/RANGEEDGES"
            f"/T/(Jan%20{year})/({last}%20{year})/RANGEEDGES/data.nc"
        )
        content = _get(url, timeout=600).content
        if not content.startswith(b"CDF"):
            raise RuntimeError(f"Respuesta inesperada de IRI para {year}")
        dest.write_bytes(content)
        log.info("CHIRPS %d: %.1f MB", year, len(content) / 1e6)
    return cache


def download_all(force: bool = False) -> None:
    from .clean import load_municipios

    download_eva(force)
    divipola = download_divipola(force)
    download_oni(force)
    download_chirps(force)
    download_power(load_municipios(divipola))
