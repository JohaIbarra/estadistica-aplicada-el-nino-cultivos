"""Carga de artefactos para el dashboard (solo lectura, con caché)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))  # permite ejecutar sin instalar el paquete

REQUIRED = [
    "panel_app",
    "municipios",
    "enso_anual",
    "anomalia_anual",
    "efectos_cultivo",
    "efectos_fase",
    "efectos_region",
    "canal_climatico",
    "lluvia_vs_oni",
    "robustez",
    "backtest_metricas",
    "backtest_global",
    "calibracion",
    "shap_global",
    "pronostico",
    "bitacora",
    "control_negativo",
]


def _st_cache(fn):
    try:
        import streamlit as st

        return st.cache_data(show_spinner=False)(fn)
    except Exception:  # pragma: no cover - fuera de streamlit
        return fn


@_st_cache
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


@_st_cache
def summary() -> dict:
    return json.loads((PROCESSED / "resumen.json").read_text(encoding="utf-8"))


@_st_cache
def model_meta() -> dict:
    return json.loads((PROCESSED / "modelo_meta.json").read_text(encoding="utf-8"))


def load_model():
    import lightgbm as lgb

    return lgb.Booster(model_file=str(PROCESSED / "modelo_lgbm.txt"))


def check_artifacts() -> None:
    """Falla con un mensaje claro si falta algún artefacto (usado en CI)."""
    missing = [n for n in REQUIRED if not (PROCESSED / f"{n}.parquet").exists()]
    for f in ("resumen.json", "modelo_meta.json", "modelo_lgbm.txt"):
        if not (PROCESSED / f).exists():
            missing.append(f)
    if missing:
        raise SystemExit(f"Faltan artefactos en data/processed: {missing}. Ejecute `agroclima all`.")
    print(f"OK: {len(REQUIRED) + 3} artefactos presentes")


if __name__ == "__main__":
    check_artifacts()
