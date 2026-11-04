"""Orquestación: datos crudos -> panel -> inferencia -> pronóstico -> artefactos.

Cada etapa lee de disco lo que produjo la anterior, así que se puede
reejecutar cualquier etapa por separado (``agroclima <etapa>``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, climate, download, enso, inference, panel, predict
from . import config as C

log = logging.getLogger(__name__)


def _save(df: pd.DataFrame, name: str) -> Path:
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    path = C.PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    return path


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / f"{name}.parquet")


def _timed(fn):
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        log.info("%s: %.1f s", fn.__name__, time.perf_counter() - t0)
        return out

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# --------------------------------------------------------------------------
@_timed
def stage_build() -> pd.DataFrame:
    """Limpieza, clima y panel analítico."""
    municipios = clean.load_municipios()
    raw = clean.load_eva()
    eva, log_clean = clean.clean_eva(raw)
    annual = clean.to_annual(eva)
    filtered, log_panel = panel.filter_panel(annual, clean.flag_outliers(annual))

    power, elev = download.load_power_cache()
    monthly = climate.municipal_monthly(
        climate.chirps_municipal(municipios), climate.power_municipal(power, municipios)
    )
    clima = climate.annual_features(climate.add_anomalies(monthly))
    elevacion = climate.municipal_elevation(elev, municipios)
    enso_annual = enso.annual_exposure(enso.load_oni())

    P = panel.build_panel(filtered, enso_annual, clima, municipios, elevacion)

    names = annual.groupby("cod_mun")[["municipio", "departamento"]].last().reset_index()
    mun_out = municipios.merge(names, on="cod_mun", how="left").merge(elevacion, on="cod_mun", how="left")
    mun_out["region"] = mun_out["cod_mun"].str[:2].map(C.DEPTO_A_REGION)

    _save(P, "panel")
    app_cols = [
        "cod_mun",
        "municipio",
        "departamento",
        "region",
        "cultivo",
        "variante",
        "anio",
        "rendimiento",
        "area_cosechada",
        "produccion",
        "anom_pct",
        "repetido",
        "lat",
        "lon",
    ]
    _save(P[app_cols], "panel_app")
    _save(clima, "clima_municipal")
    _save(enso_annual, "enso_anual")
    _save(mun_out, "municipios")
    _save(
        pd.concat([log_clean.assign(etapa="registros EVA"), log_panel.assign(etapa="panel anual")]),
        "bitacora",
    )
    return P


@_timed
def stage_inference() -> dict:
    """Efecto de ENSO: por cultivo, por fase, por región, canal climático y robustez."""
    P = _load("panel")
    enso_annual = _load("enso_anual")
    clima = _load("clima_municipal")

    effects = inference.effects_by_crop(P)
    phases = inference.phase_contrasts(P)
    het = inference.heterogeneity(P)
    channel = inference.climate_channel(P)
    rain_map = inference.enso_rainfall_map(clima, enso_annual)
    robust = inference.robustness(P)

    for df, name in (
        (effects, "efectos_cultivo"),
        (phases, "efectos_fase"),
        (het, "efectos_region"),
        (channel, "canal_climatico"),
        (rain_map, "lluvia_vs_oni"),
        (robust, "robustez"),
    ):
        _save(df, name)

    national = effects.iloc[0]
    yearly = (
        P.groupby("anio")
        .agg(anom=("anom_log_rend", "mean"), oni=("oni_anual", "first"), fase=("fase", "first"))
        .reset_index()
    )
    yearly["anom_pct"] = 100 * (np.exp(yearly["anom"]) - 1)
    _save(yearly, "anomalia_anual")
    return {
        "efecto_nacional_pct": float(national["efecto_pct"]),
        "efecto_nacional_ic": [float(national["ic_low_pct"]), float(national["ic_high_pct"])],
        "p_permutacion": float(national["p_permutacion"]),
        "ratio_se_bootstrap_vs_ingenuo": float(national["se_bootstrap"] / national["se_ingenuo"]),
    }


@_timed
def stage_forecast() -> dict:
    """Backtest temporal, modelo final, SHAP y pronóstico para el año objetivo."""
    P = _load("panel")
    enso_annual = _load("enso_anual")
    clima = _load("clima_municipal")
    target_years = list(range(C.YEAR_MIN + predict.LAG + 4, C.FORECAST_YEAR + 1))
    ex = predict.build_examples(P, enso_annual, clima, target_years)

    metrics, preds = predict.backtest(ex)
    preds["p_cal"] = predict.calibrate_backtest(preds)
    calibrator = predict.fit_calibrator(preds)
    pooled = predict.pooled_metrics(preds)
    calib = predict.calibration_table(preds)
    control = predict.negative_control(ex, within=["anio"])
    control_mun = predict.negative_control(ex, within=["cod_mun", "anio"])
    log.info("Control negativo: %s | dentro de municipio-año: %s", control, control_mun)

    # La calibración se adopta solo si mejora el Brier del backtest (decisión guiada por datos).
    brier = pooled.set_index("modelo")["brier"]
    use_cal = bool(brier["LightGBM calibrado"] < brier["LightGBM"])
    chosen = "LightGBM calibrado" if use_cal else "LightGBM"
    log.info(
        "Calibración isotónica %s (Brier %.4f vs %.4f)",
        "adoptada" if use_cal else "descartada",
        brier["LightGBM calibrado"],
        brier["LightGBM"],
    )

    labelled = ex.dropna(subset=["mala_cosecha"])
    model = predict.fit_lgbm(labelled)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(C.PROCESSED / "modelo_lgbm.txt"))

    future = ex[ex["anio"] == C.FORECAST_YEAR].copy()
    future = future[future["anios_historia"] >= 5]
    future["prob_cruda"] = model.predict_proba(future[predict.FEATURES])[:, 1]
    future["prob_mala_cosecha"] = (
        predict.apply_calibrator(future["prob_cruda"].to_numpy(), calibrator)
        if use_cal
        else future["prob_cruda"]
    )
    # Contrafactual: mismo año, mismas series, pero con ENSO neutral (ONI 0, sin tendencia).
    neutral = future.copy()
    neutral["oni_jas_prev"], neutral["oni_tend_prev"] = 0.0, 0.0
    p_neutral = model.predict_proba(neutral[predict.FEATURES])[:, 1]
    future["prob_neutral"] = predict.apply_calibrator(p_neutral, calibrator) if use_cal else p_neutral
    enso_shift = (
        future.assign(
            ha=future["prob_mala_cosecha"] * future["area_ref"],
            ha0=future["prob_neutral"] * future["area_ref"],
        )
        .groupby("cultivo", as_index=False)
        .agg(area=("area_ref", "sum"), ha=("ha", "sum"), ha_neutral=("ha0", "sum"))
    )
    enso_shift["cambio_pp"] = 100 * (enso_shift["ha"] - enso_shift["ha_neutral"]) / enso_shift["area"]
    _save(enso_shift, "pronostico_vs_neutral")

    contrib = predict.shap_values(model, future)
    top = contrib.apply(lambda r: list(r.abs().sort_values(ascending=False).index[:3]), axis=1)
    future["factores_principales"] = top.map(lambda xs: ", ".join(xs))
    future["contrib_oni"] = contrib["oni_jas_prev"] + contrib["oni_tend_prev"]

    sample = labelled.sample(min(20000, len(labelled)), random_state=C.RANDOM_SEED)
    shap_sample = predict.shap_values(model, sample)
    shap_global = (
        shap_sample.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("shap_medio_abs")
        .rename_axis("variable")
        .reset_index()
    )
    dep = sample[["oni_jas_prev", "cultivo"]].assign(shap_oni=shap_sample["oni_jas_prev"].to_numpy())

    for c in predict.CATEGORICAL:
        future[c] = future[c].astype(str)
        ex[c] = ex[c].astype(str)
    meta = {
        "categorias": {c: list(labelled[c].cat.categories) for c in predict.CATEGORICAL},
        "features": predict.FEATURES,
        "anio_pronostico": C.FORECAST_YEAR,
        "oni_jas_prev": float(future["oni_jas_prev"].iloc[0]),
        "oni_tend_prev": float(future["oni_tend_prev"].iloc[0]),
        # Rango del ONI visto en entrenamiento: los árboles no extrapolan fuera de él.
        "oni_entrenamiento": [float(labelled["oni_jas_prev"].min()), float(labelled["oni_jas_prev"].max())],
        "calibrador": calibrator if use_cal else None,
        "modelo_elegido": chosen,
    }
    (C.PROCESSED / "modelo_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _save(
        pd.DataFrame(
            [
                {"control": "Etiquetas permutadas dentro de cada año", **control},
                {"control": "Etiquetas permutadas dentro de cada municipio-año", **control_mun},
            ]
        ),
        "control_negativo",
    )
    _save(metrics, "backtest_metricas")
    _save(
        preds.assign(variante=preds["variante"].astype(str), region=preds["region"].astype(str)),
        "backtest_predicciones",
    )
    _save(pooled, "backtest_global")
    _save(calib, "calibracion")
    _save(shap_global, "shap_global")
    _save(dep, "shap_oni")
    _save(future, "pronostico")

    lg = pooled.set_index("modelo").loc[chosen]
    return {
        "anio_pronostico": C.FORECAST_YEAR,
        "roc_auc": float(lg["roc_auc"]),
        "pr_auc": float(lg["pr_auc"]),
        "brier_skill": float(lg["brier_skill"]),
        "precision_top10": float(lg["precision_top10"]),
        "pr_auc_ic": [float(lg["pr_auc_low"]), float(lg["pr_auc_high"])],
        "precision_top10_ic": [float(lg["precision_top10_low"]), float(lg["precision_top10_high"])],
        "modelo_elegido": chosen,
        "control_negativo_roc_auc": float(control["roc_auc"]),
        "control_municipio_anio_roc_auc": float(control_mun["roc_auc"]),
        "control_negativo_pr_auc": float(control["pr_auc"]),
        "tasa_base": float(lg["tasa_base"]),
        "series_pronosticadas": int(len(future)),
        "hectareas_en_riesgo": float((future["prob_mala_cosecha"] * future["area_ref"]).sum()),
        "hectareas_en_riesgo_neutral": float((future["prob_neutral"] * future["area_ref"]).sum()),
    }


def run(stages: list[str]) -> None:
    summary_path = C.PROCESSED / "resumen.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    def save_summary() -> None:
        C.PROCESSED.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if "download" in stages:
        download.download_all()
    if "build" in stages:
        P = stage_build()
        summary["panel"] = {
            "filas": int(len(P)),
            "series": int(P["serie"].nunique()),
            "municipios": int(P["cod_mun"].nunique()),
            "anios": [int(P["anio"].min()), int(P["anio"].max())],
            "frac_repetidos": float(P["repetido"].mean()),
            "frac_enteros": float(np.isclose(P["rendimiento"], P["rendimiento"].round()).mean()),
        }
        save_summary()
    if "inference" in stages:
        summary["inferencia"] = stage_inference()
        save_summary()
    if "forecast" in stages:
        summary["pronostico"] = stage_forecast()
        save_summary()
    if "report" in stages:
        from . import report

        report.build_all()
