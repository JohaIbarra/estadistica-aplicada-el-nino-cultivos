"""Alerta temprana de malas cosechas, emitida el 1 de octubre del año anterior.

Pregunta operativa: el 1 de octubre de t-1, con la información publicada a esa
fecha, ¿qué municipio-cultivo tendrá en t un rendimiento al menos 20 % por
debajo de su nivel reciente?

Reglas contra la fuga de información (cada una está cubierta por un test):

1. Rendimientos: solo hasta t-2. La EVA del año t-1 aún no está publicada el
   1 de octubre de t-1.
2. ONI: solo julio-septiembre de t-1. Clima: solo enero-septiembre de t-1.
3. Validación temporal: para evaluar el año T se entrena únicamente con años
   objetivo <= T-2, cuyos resultados ya se conocían al emitir el pronóstico de T.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

LAG = C.EVA_PUBLICATION_LAG
CATEGORICAL = ["variante", "region"]
NUMERIC = [
    "oni_jas_prev",
    "oni_tend_prev",
    "lluvia_anom_pct_prev",
    "temp_anom_prev",
    "tmax_anom_prev",
    "meses_secos_prev",
    "meses_humedos_prev",
    "desvio_ultimo",
    "volatilidad",
    "tendencia",
    "tasa_malas_hist",
    "mala_ultima",
    "frac_repetidos",
    "anios_historia",
    "log_area",
    "mezcla_metodologia",
    "lat",
    "lon",
    "elevacion_m",
]
FEATURES = CATEGORICAL + NUMERIC
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


# --------------------------------------------------------------------------
# Construcción de ejemplos (serie x año objetivo) sin mirar el futuro
# --------------------------------------------------------------------------
def _wide(panel: pd.DataFrame, col: str, years: np.ndarray) -> pd.DataFrame:
    return panel.pivot_table(index="serie", columns="anio", values=col, aggfunc="first").reindex(
        columns=years
    )


def history_features(panel: pd.DataFrame, target_years: list[int]) -> pd.DataFrame:
    """Rasgos de historia productiva para cada serie y año objetivo t (usa solo <= t-2)."""
    years = np.arange(C.YEAR_MIN, max(target_years) + 1)
    logy = np.log(_wide(panel, "rendimiento", years))
    area = _wide(panel, "area_cosechada", years)
    Y, A = logy.to_numpy(), area.to_numpy()
    col = {y: i for i, y in enumerate(years)}

    def baseline(arr: np.ndarray, t: int) -> np.ndarray:
        cols = [col[s] for s in range(t - LAG - C.BASELINE_YEARS + 1, t - LAG + 1) if s in col]
        block = arr[:, cols]
        enough = np.sum(np.isfinite(block), axis=1) >= 3
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(block, axis=1) if block.size else np.full(len(arr), np.nan)
        return np.where(enough, med, np.nan)

    # Malas cosechas pasadas: para cada año s, ¿y_s < baseline_s? Solo se usan s <= t-2.
    past_bad = np.full_like(Y, np.nan)
    for s in years:
        if s - LAG - C.BASELINE_YEARS + 1 < years[0]:
            continue
        b = baseline(Y, s)
        ys = Y[:, col[s]]
        ok = np.isfinite(b) & np.isfinite(ys)
        past_bad[ok, col[s]] = (ys[ok] < b[ok] + np.log(1 - C.BAD_HARVEST_DROP)).astype(float)

    frames = []
    for t in target_years:
        known = [col[s] for s in years if s <= t - LAG]
        if not known:
            continue
        Yk, Ak, Bk = Y[:, known], A[:, known], past_bad[:, known]
        base = baseline(Y, t)
        last_idx = np.where(np.isfinite(Yk), np.arange(len(known)), -1).max(axis=1)
        has = last_idx >= 0
        last = np.full(len(Y), np.nan)
        last[has] = Yk[has, last_idx[has]]
        last_area = np.full(len(Y), np.nan)
        last_area[has] = Ak[has, last_idx[has]]
        recent = Yk[:, -8:]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # series sin historia: NaN esperado
            vol = np.nanstd(recent, axis=1, ddof=1)
            yrs = np.array([years[k] for k in known], dtype=float)[-5:]
            r5 = Yk[:, -5:]
            trend = np.array(
                [
                    np.polyfit(yrs[np.isfinite(r)], r[np.isfinite(r)], 1)[0]
                    if np.isfinite(r).sum() >= 3
                    else np.nan
                    for r in r5
                ]
            )
            bad_rate = np.nanmean(Bk, axis=1)
            # Persistencia: ¿fue mala la última cosecha conocida (t-2)?
            last_bad = np.full(len(Y), np.nan)
            last_bad[has] = Bk[has, last_idx[has]]
            # Fracción de años consecutivos con rendimiento idéntico (dato arrastrado)
            pairs = np.isfinite(Yk[:, 1:]) & np.isfinite(Yk[:, :-1])
            same = np.isclose(Yk[:, 1:], Yk[:, :-1], rtol=0, atol=1e-3) & pairs
            rep_frac = same.sum(axis=1) / np.maximum(pairs.sum(axis=1), 1)
            rep_frac = np.where(pairs.sum(axis=1) > 0, rep_frac, np.nan)
        base_years = [s for s in range(t - LAG - C.BASELINE_YEARS + 1, t - LAG + 1)]
        mezcla = np.mean([s < 2019 for s in base_years]) if t >= 2019 else 0.0
        frames.append(
            pd.DataFrame(
                {
                    "serie": logy.index,
                    "anio": t,
                    "base_log": base,
                    "desvio_ultimo": last - base,
                    "volatilidad": vol,
                    "tendencia": trend,
                    "tasa_malas_hist": bad_rate,
                    "mala_ultima": last_bad,
                    "frac_repetidos": rep_frac,
                    "anios_historia": np.isfinite(Yk).sum(axis=1),
                    "log_area": np.log(last_area),
                    "area_ref": last_area,
                    "mezcla_metodologia": mezcla,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["base_log"])]


def build_examples(
    panel: pd.DataFrame,
    enso_annual: pd.DataFrame,
    clima_mun: pd.DataFrame,
    target_years: list[int],
) -> pd.DataFrame:
    """Una fila por serie y año objetivo, con la etiqueta cuando el año ya ocurrió."""
    hist = history_features(panel, target_years)
    static = panel.groupby("serie")[
        ["cod_mun", "cultivo", "variante", "region", "lat", "lon", "elevacion_m"]
    ].first()
    ex = hist.merge(static, left_on="serie", right_index=True, how="left")
    ex = ex.merge(enso_annual[["anio", "oni_jas_prev", "oni_tend_prev"]], on="anio", how="left")
    clim_cols = [c for c in clima_mun.columns if c.endswith("_prev")]
    ex = ex.merge(clima_mun[["cod_mun", "anio", *clim_cols]], on=["cod_mun", "anio"], how="left")
    obs = panel[["serie", "anio", "rendimiento"]]
    ex = ex.merge(obs, on=["serie", "anio"], how="left")
    ex["mala_cosecha"] = np.where(
        ex["rendimiento"].notna(),
        (np.log(ex["rendimiento"]) < ex["base_log"] + np.log(1 - C.BAD_HARVEST_DROP)).astype(float),
        np.nan,
    )
    for c in NUMERIC:
        if c not in ex:
            ex[c] = np.nan  # p. ej. sin clima disponible: LightGBM maneja faltantes
    for c in CATEGORICAL:
        ex[c] = ex[c].astype("category")
    return ex


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def fit_lgbm(train: pd.DataFrame, seed: int = C.RANDOM_SEED) -> LGBMClassifier:
    model = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    model.fit(train[FEATURES], train["mala_cosecha"].astype(int), categorical_feature=CATEGORICAL)
    return model


def fit_logistic(train: pd.DataFrame):
    cols = ["oni_jas_prev", "lluvia_anom_pct_prev", "volatilidad", "tasa_malas_hist", "desvio_ultimo"]
    fill = train[cols].median().fillna(0.0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[cols].fillna(fill), train["mala_cosecha"].astype(int))
    model.cols_, model.fill_ = cols, fill
    return model


def predict_logistic(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[model.cols_].fillna(model.fill_))[:, 1]


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Referencia ingenua: frecuencia histórica de malas cosechas del cultivo."""
    rate = train.groupby("variante", observed=True)["mala_cosecha"].mean()
    return (
        test["variante"]
        .astype(str)
        .map(rate.rename(index=str))
        .fillna(train["mala_cosecha"].mean())
        .to_numpy()
    )


def persistence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Referencia de persistencia: "si la última cosecha conocida fue mala, esta también".

    Se expresa como probabilidad con las frecuencias de entrenamiento de cada caso
    (última mala / no mala / sin dato), para que también tenga Brier.
    """
    key_tr = train["mala_ultima"].fillna(-1)
    rate = train.groupby(key_tr)["mala_cosecha"].mean()
    return test["mala_ultima"].fillna(-1).map(rate).fillna(train["mala_cosecha"].mean()).to_numpy()


def _at_budget(y: np.ndarray, p: np.ndarray, budget: float) -> tuple[float, float]:
    """Precisión y exhaustividad si solo se pueden atender ``budget`` de los casos."""
    k = max(1, int(round(budget * len(p))))
    order = np.argsort(-p, kind="stable")[:k]
    return float(y[order].mean()), float(y[order].sum() / max(1, y.sum()))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p10, r10 = _at_budget(y, p, 0.10)
    p20, r20 = _at_budget(y, p, 0.20)
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "brier": brier_score_loss(y, p),
        "precision_top10": p10,
        "recall_top10": r10,
        "precision_top20": p20,
        "recall_top20": r20,
        "tasa_base": float(y.mean()),
    }


MODEL_COLS = {
    "Climatología (referencia)": "p_clim",
    "Persistencia (referencia)": "p_pers",
    "Regresión logística": "p_logit",
    "LightGBM": "p_lgbm",
    "LightGBM calibrado": "p_cal",
}


def backtest(
    examples: pd.DataFrame, first_test: int = C.FIRST_TEST_YEAR, baselines: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validación con origen móvil. Devuelve (métricas por año y modelo, predicciones)."""
    lab = examples.dropna(subset=["mala_cosecha"])
    preds, rows = [], []
    for T in sorted(y for y in lab["anio"].unique() if y >= first_test):
        train = lab[lab["anio"] <= T - LAG]
        test = lab[lab["anio"] == T]
        if len(test) == 0 or train["anio"].nunique() < 3:
            continue
        p = {"LightGBM": fit_lgbm(train).predict_proba(test[FEATURES])[:, 1]}
        if baselines:
            p["Climatología (referencia)"] = climatology(train, test)
            p["Persistencia (referencia)"] = persistence(train, test)
            p["Regresión logística"] = predict_logistic(fit_logistic(train), test)
        y = test["mala_cosecha"].to_numpy()
        for name, prob in p.items():
            rows.append({"anio": T, "modelo": name, "n": len(test), **_metrics(y, prob)})
        preds.append(
            test[["serie", "cod_mun", "anio", "cultivo", "variante", "region", "mala_cosecha"]].assign(
                **{MODEL_COLS[name]: prob for name, prob in p.items()}
            )
        )
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def calibrate_backtest(preds: pd.DataFrame) -> pd.Series:
    """Calibración isotónica sin mirar el futuro.

    La probabilidad del año T se recalibra con un mapa aprendido solo sobre las
    predicciones de backtest de años <= T-2 (ya evaluables al emitir T). Los
    primeros años, sin historia suficiente, conservan la probabilidad cruda.
    """
    out = preds["p_lgbm"].copy()
    for T in sorted(preds["anio"].unique()):
        past = preds[preds["anio"] <= T - LAG]
        if past["anio"].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(past["p_lgbm"], past["mala_cosecha"])
        mask = preds["anio"] == T
        out[mask] = iso.predict(preds.loc[mask, "p_lgbm"])
    return out


def fit_calibrator(preds: pd.DataFrame) -> dict:
    """Mapa isotónico final (todas las predicciones de backtest) como tabla interpolable."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(preds["p_lgbm"], preds["mala_cosecha"])
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_calibrator(p: np.ndarray, cal: dict) -> np.ndarray:
    return np.interp(p, cal["x"], cal["y"])


def grouped_bootstrap_ci(
    preds: pd.DataFrame, col: str, group: str = "cod_mun", n_boot: int = 300, seed: int = C.RANDOM_SEED
) -> dict:
    """IC 95 % remuestreando municipios completos (sus series comparten clima y encuestador)."""
    rng = np.random.default_rng(seed)
    _, idx = np.unique(preds[group].to_numpy(), return_inverse=True)
    order = np.argsort(idx, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(idx[order]) != 0, True])
    members = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    y_all, p_all = preds["mala_cosecha"].to_numpy(), preds[col].to_numpy()
    stats: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "precision_top10": []}
    for _ in range(n_boot):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        y, p = y_all[pick], p_all[pick]
        stats["pr_auc"].append(average_precision_score(y, p))
        stats["roc_auc"].append(roc_auc_score(y, p))
        stats["precision_top10"].append(_at_budget(y, p, 0.10)[0])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}


def pooled_metrics(preds: pd.DataFrame, ci: bool = True) -> pd.DataFrame:
    y = preds["mala_cosecha"].to_numpy()
    rows = []
    for name, col in MODEL_COLS.items():
        if col not in preds:
            continue
        row = {"modelo": name, **_metrics(y, preds[col].to_numpy())}
        if ci:
            for k, (lo, hi) in grouped_bootstrap_ci(preds, col).items():
                row[f"{k}_low"], row[f"{k}_high"] = lo, hi
        rows.append(row)
    out = pd.DataFrame(rows)
    ref = out.loc[out["modelo"] == "Climatología (referencia)", "brier"].iloc[0]
    out["brier_skill"] = 1 - out["brier"] / ref
    return out


def negative_control(examples: pd.DataFrame, within: list[str], seed: int = C.RANDOM_SEED) -> dict:
    """Control negativo: permuta las etiquetas dentro de ``within`` y repite el backtest.

    - ``within=["anio"]``: rompe todo vínculo entre las variables y la etiqueta
      salvo la prevalencia anual. Sin fugas, ROC-AUC ~ 0.5 y PR-AUC ~ tasa base.
    - ``within=["cod_mun", "anio"]``: conserva la prevalencia de cada
      municipio-año, así que el clima local y el ONI aún pueden predecirla
      legítimamente. Mide cuánta señal vive a nivel municipio-año y no en la
      historia de cada serie.
    """
    rng = np.random.default_rng(seed)
    ex = examples.copy()
    lab = ex["mala_cosecha"].notna()
    ex.loc[lab, "mala_cosecha"] = (
        ex.loc[lab].groupby(within)["mala_cosecha"].transform(lambda s: rng.permutation(s.to_numpy()))
    )
    _, preds = backtest(ex, baselines=False)
    return _metrics(preds["mala_cosecha"].to_numpy(), preds["p_lgbm"].to_numpy())


def calibration_table(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Deciles de probabilidad vs. frecuencia observada, cruda y calibrada."""
    frames = []
    for col, label in (("p_lgbm", "Cruda"), ("p_cal", "Calibrada")):
        if col not in preds:
            continue
        q = pd.qcut(preds[col].rank(method="first"), bins)
        t = (
            preds.groupby(q, observed=True)
            .agg(prob_media=(col, "mean"), frecuencia_observada=("mala_cosecha", "mean"), n=(col, "size"))
            .reset_index(drop=True)
        )
        frames.append(t.assign(version=label))
    return pd.concat(frames, ignore_index=True)


def shap_values(model: LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones SHAP exactas para árboles (TreeSHAP), en escala log-odds."""
    contrib = model.booster_.predict(X[FEATURES], pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X.index)
