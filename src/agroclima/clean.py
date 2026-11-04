"""Armonización y limpieza de las Evaluaciones Agropecuarias Municipales (EVA).

Problemas reales que resuelve este módulo:

1. Dos bases con esquemas distintos (2007-2018 y 2019-2025): columnas,
   mayúsculas, tildes y desagregaciones cambian entre ambas.
2. Códigos DANE sin el cero inicial (``5001`` en vez de ``05001``).
3. Variedades que se dividen en la base nueva (maíz amarillo / blanco) y que
   hay que reagrupar para que la serie sea comparable en el tiempo.
4. Cultivos transitorios reportados por semestre (A/B) y permanentes por año.
5. Registros imposibles (rendimiento cero con producción positiva, áreas
   cosechadas mayores a las sembradas, valores atípicos de digitación).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

HIST_COLS = {
    "c_d_dep": "cod_dep",
    "c_d_mun": "cod_mun",
    "municipio": "municipio",
    "departamento": "departamento",
    "cultivo": "cultivo_raw",
    "desagregaci_n_regional_y": "variante_raw",
    "a_o": "anio",
    "periodo": "periodo",
    "rea_sembrada_ha": "area_sembrada",
    "rea_cosechada_ha": "area_cosechada",
    "producci_n_t": "produccion",
}
REC_COLS = {
    "c_digo_dane_departamento": "cod_dep",
    "c_digo_dane_municipio": "cod_mun",
    "municipio": "municipio",
    "departamento": "departamento",
    "cultivo": "cultivo_raw",
    "desagregaci_n_cultivo": "variante_raw",
    "a_o": "anio",
    "periodo": "periodo",
    "rea_sembrada": "area_sembrada",
    "rea_cosechada": "area_cosechada",
    "producci_n": "produccion",
}


def normalize_text(s: pd.Series) -> pd.Series:
    """MAYÚSCULAS sin tildes ni espacios repetidos; conserva la Ñ como N."""

    def _norm(x: object) -> object:
        if not isinstance(x, str):
            return x
        x = unicodedata.normalize("NFKD", x)
        x = "".join(ch for ch in x if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", x).strip().upper()

    return s.map(_norm)


def harmonize_crop(cultivo: str, variante: str) -> tuple[str | None, str | None]:
    """Devuelve (cultivo canónico, variante comparable 2007-2025) o (None, None).

    Las variantes separan sistemas productivos con rendimientos muy distintos
    (p. ej. maíz tecnificado ~5 t/ha vs tradicional ~1.8 t/ha). Mezclarlas
    convertiría cambios en la composición del área en falsos cambios de
    rendimiento.
    """
    c, v = cultivo or "", variante or ""
    if c == "CAFE":
        return "CAFE", "CAFE"
    if c == "CACAO":
        return "CACAO", "CACAO"
    if c == "PALMA DE ACEITE":
        return "PALMA DE ACEITE", "PALMA DE ACEITE"
    if c in {"FRIJOL", "FRÍJOL"}:
        return "FRIJOL", "FRIJOL"  # la base nueva ya no separa arbustivo/voluble
    if c == "AGUACATE":
        return "AGUACATE", "AGUACATE"  # la base histórica no separa Hass
    if c == "ARROZ":
        for k in ("RIEGO", "SECANO MECANIZADO", "SECANO MANUAL"):
            if k in v:
                return "ARROZ", f"ARROZ {k}"
        return None, None
    if c == "MAIZ" and "FORRAJERO" not in v:
        if "TECNIFICADO" in v:
            return "MAIZ", "MAIZ TECNIFICADO"
        if "TRADICIONAL" in v:
            return "MAIZ", "MAIZ TRADICIONAL"
        return None, None
    if c == "PAPA":
        return ("PAPA", "PAPA CRIOLLA") if "CRIOLLA" in v else ("PAPA", "PAPA")
    if c == "PLATANO":
        if "EXPORTACION" in v:
            return "PLATANO", "PLATANO EXPORTACION"
        if v in {"PLATANO", "PLATANO CONSUMO INTERNO"}:
            return "PLATANO", "PLATANO"
        return None, None
    if c == "YUCA":
        return ("YUCA", "YUCA INDUSTRIAL") if "INDUSTRIAL" in v else ("YUCA", "YUCA")
    if c == "BANANO":
        return ("BANANO", "BANANO EXPORTACION") if "EXPORTACION" in v else ("BANANO", "BANANO")
    if c == "CANA PANELERA" or (c == "CANA" and v == "CANA PANELERA"):
        return "CANA PANELERA", "CANA PANELERA"
    return None, None


def _standardize(df: pd.DataFrame, mapping: dict[str, str], fuente: str) -> pd.DataFrame:
    out = df[list(mapping)].rename(columns=mapping).copy()
    out["fuente"] = fuente
    return out


def load_eva(hist_csv: Path | None = None, rec_csv: Path | None = None) -> pd.DataFrame:
    hist_csv = hist_csv or C.RAW / "eva_2007_2018.csv"
    rec_csv = rec_csv or C.RAW / "eva_2019_2025.csv"
    hist = _standardize(pd.read_csv(hist_csv, dtype=str), HIST_COLS, "EVA 2007-2018")
    rec = _standardize(pd.read_csv(rec_csv, dtype=str), REC_COLS, "EVA 2019-2025")
    return pd.concat([hist, rec], ignore_index=True)


def clean_eva(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia registros semestrales/anuales. Devuelve (datos, bitácora de descartes)."""
    df = raw.copy()
    log: list[dict] = [{"paso": "registros originales", "filas": len(df)}]

    df["cod_mun"] = df["cod_mun"].str.strip().str.zfill(5)
    df["cod_dep"] = df["cod_mun"].str[:2]
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    for col in ("area_sembrada", "area_cosechada", "produccion"):
        df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
    df["municipio"] = normalize_text(df["municipio"])
    df["departamento"] = normalize_text(df["departamento"])
    c_norm = normalize_text(df["cultivo_raw"])
    v_norm = normalize_text(df["variante_raw"])
    pairs = [harmonize_crop(c, v) for c, v in zip(c_norm, v_norm, strict=True)]
    df["cultivo"] = [p[0] for p in pairs]
    df["variante"] = [p[1] for p in pairs]

    def _drop(mask: pd.Series, motivo: str) -> None:
        nonlocal df
        log.append({"paso": motivo, "filas": int(mask.sum())})
        df = df[~mask]

    _drop(df["cultivo"].isna(), "cultivo fuera del alcance del estudio")
    _drop(~df["anio"].between(C.YEAR_MIN, C.YEAR_MAX), f"año fuera de {C.YEAR_MIN}-{C.YEAR_MAX}")
    _drop(df[["area_cosechada", "produccion"]].isna().any(axis=1), "área o producción faltante")
    _drop((df["area_cosechada"] <= 0) | (df["produccion"] <= 0), "área o producción en cero (sin cosecha)")
    _drop(
        df["area_sembrada"].notna() & (df["area_cosechada"] > 1.05 * df["area_sembrada"]),
        "área cosechada mayor que la sembrada",
    )
    df = df.drop(columns=["cultivo_raw", "variante_raw"])
    return df.reset_index(drop=True), pd.DataFrame(log)


def to_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega semestres A/B al año: rendimiento = producción total / área total."""
    keys = ["cod_mun", "cod_dep", "cultivo", "variante", "anio"]
    names = df.groupby("cod_mun")[["municipio", "departamento"]].last()
    agg = df.groupby(keys, as_index=False).agg(
        area_cosechada=("area_cosechada", "sum"),
        produccion=("produccion", "sum"),
        fuente=("fuente", "first"),
    )
    agg["rendimiento"] = agg["produccion"] / agg["area_cosechada"]
    agg = agg.merge(names, left_on="cod_mun", right_index=True, how="left")
    agg["region"] = agg["cod_dep"].map(C.DEPTO_A_REGION)
    agg["serie"] = agg["cod_mun"] + "|" + agg["variante"]
    return agg


def flag_outliers(annual: pd.DataFrame, k: float = C.OUTLIER_MAD, min_factor: float = 3.0) -> pd.Series:
    """Marca errores de registro: lejos de la mediana de su serie en MAD *y* en magnitud.

    Dos condiciones a la vez:

    - más de ``k`` MAD robustas de la mediana de la serie (en log), y
    - al menos ``min_factor`` veces por encima o por debajo de esa mediana.

    La segunda evita borrar malas cosechas reales: en series muy estables la MAD
    es pequeña y una caída genuina de 30 % ya supera 5 MAD. Los errores de
    digitación típicos (coma corrida, toneladas por kilos) son de x10 o más.
    """
    logy = np.log(annual["rendimiento"])
    med = logy.groupby(annual["serie"]).transform("median")
    dev = (logy - med).abs()
    mad = dev.groupby(annual["serie"]).transform("median")
    mad = mad.where(mad > 0.05, 0.05)  # piso: series casi constantes
    return (dev / (1.4826 * mad) > k) & (dev > np.log(min_factor))


# Caja que contiene a Colombia continental e insular (San Andrés incluido).
LAT_RANGE, LON_RANGE = (-4.3, 13.6), (-82.0, -66.8)


def repair_coordinate(value: float, lo: float, hi: float) -> float:
    """Recupera coordenadas a las que se les perdió el punto decimal.

    En la DIVIPOLA publicada hay municipios con ``525`` en vez de ``5.25``. Si
    al dividir por potencias de 10 el valor cae en el rango válido, se acepta;
    si no, se devuelve NaN para no inventar una ubicación.
    """
    if np.isnan(value):
        return np.nan
    for p in range(0, 12):
        v = value / 10**p
        if lo <= v <= hi:
            return v
    return np.nan


def load_municipios(path: Path | None = None) -> pd.DataFrame:
    path = path or C.RAW / "divipola_municipios.csv"
    m = pd.read_csv(path, dtype=str)
    out = pd.DataFrame(
        {
            "cod_mun": m["idmupio"].str.zfill(5),
            "nombre": m["nommpio"],
            "lat": m["mpiolatitud"].astype(float).map(lambda v: repair_coordinate(v, *LAT_RANGE)),
            "lon": m["mpiolongitud"].astype(float).map(lambda v: repair_coordinate(v, *LON_RANGE)),
        }
    )
    return out.dropna(subset=["lat", "lon"]).drop_duplicates("cod_mun").reset_index(drop=True)
