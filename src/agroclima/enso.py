"""Índice Oceánico El Niño (ONI) y exposición anual a ENSO.

El ONI es la media móvil de 3 meses de la anomalía de temperatura superficial
del mar en la región Niño 3.4. NOAA declara un episodio cuando el índice supera
±0.5 °C durante al menos 5 trimestres móviles consecutivos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

MISSING = -99.9


def parse_oni(text: str) -> pd.DataFrame:
    """Convierte el formato PSL (año + 12 columnas) a formato largo anio/mes/oni."""
    rows = []
    lines = text.strip().splitlines()
    first, last = (int(x) for x in lines[0].split()[:2])
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        if not first <= year <= last:
            continue
        for month, raw in enumerate(parts[1:], start=1):
            val = float(raw)
            rows.append((year, month, np.nan if val <= MISSING else val))
    df = pd.DataFrame(rows, columns=["anio", "mes", "oni"])
    return df.dropna().reset_index(drop=True)


def load_oni(path: Path | None = None) -> pd.DataFrame:
    path = path or C.RAW / "oni.data"
    return parse_oni(path.read_text(encoding="utf-8"))


def official_episodes(oni: pd.DataFrame, threshold: float = C.ONI_THRESHOLD, min_run: int = 5) -> pd.Series:
    """Fase oficial NOAA por mes: 'Niño', 'Niña' o 'Neutral'.

    Un mes pertenece a un episodio si forma parte de una racha de al menos
    ``min_run`` meses consecutivos por encima (o por debajo) del umbral.
    """
    oni = oni.sort_values(["anio", "mes"]).reset_index(drop=True)
    phase = pd.Series("Neutral", index=oni.index, dtype=object)
    for label, mask in (("Niño", oni["oni"] >= threshold), ("Niña", oni["oni"] <= -threshold)):
        run_id = (mask != mask.shift()).cumsum()
        run_len = mask.groupby(run_id).transform("size")
        phase[mask & (run_len >= min_run)] = label
    return phase


def annual_exposure(oni: pd.DataFrame) -> pd.DataFrame:
    """Resume el ONI mensual en variables anuales con distintas ventanas.

    - ``oni_anual``: media enero-diciembre del año t (exposición contemporánea,
      usada para *inferencia* del efecto).
    - ``oni_jul_jun``: media julio(t-1)-junio(t), ventana alternativa para
      análisis de sensibilidad (cultivos perennes florecen antes de cosechar).
    - ``oni_jas_prev``: media julio-septiembre de t-1 y ``oni_tend_prev``
      (septiembre menos junio de t-1, si El Niño se intensifica o se disipa). Es
      lo que se conoce el 1 de octubre de t-1, fecha de emisión del *pronóstico*.
    - ``meses_nino`` / ``meses_nina``: meses del año t dentro de un episodio oficial.
    """
    df = oni.sort_values(["anio", "mes"]).copy()
    df["fase_oficial"] = official_episodes(df).to_numpy()
    df["fecha"] = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1))
    s = df.set_index("fecha")["oni"]

    years = range(int(df["anio"].min()) + 1, int(df["anio"].max()) + 2)
    out = []
    for y in years:
        cur = s[f"{y}-01" : f"{y}-12"]
        jul_jun = s[f"{y - 1}-07" : f"{y}-06"]
        jas = s[f"{y - 1}-07" : f"{y - 1}-09"]
        ph = df.loc[df["anio"] == y, "fase_oficial"]
        out.append(
            {
                "anio": y,
                "oni_anual": cur.mean() if len(cur) == 12 else np.nan,
                "oni_jul_jun": jul_jun.mean() if len(jul_jun) == 12 else np.nan,
                "oni_jas_prev": jas.mean() if len(jas) == 3 else np.nan,
                "oni_tend_prev": (jas.iloc[-1] - s.get(pd.Timestamp(f"{y - 1}-06-01"), np.nan))
                if len(jas) == 3
                else np.nan,
                "meses_nino": int((ph == "Niño").sum()),
                "meses_nina": int((ph == "Niña").sum()),
            }
        )
    res = pd.DataFrame(out)
    res["fase"] = classify_phase(res["oni_anual"])
    return res


def classify_phase(oni_value: pd.Series, threshold: float = C.ONI_THRESHOLD) -> pd.Series:
    """Fase anual a partir de un ONI medio: Niño / Niña / Neutral (NaN si falta)."""
    phase = np.select(
        [oni_value >= threshold, oni_value <= -threshold, oni_value.notna()],
        ["Niño", "Niña", "Neutral"],
        default=None,
    )
    return pd.Series(phase, index=oni_value.index, dtype=object)
