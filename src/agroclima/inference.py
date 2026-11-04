"""Inferencia del efecto de ENSO sobre el rendimiento agrícola.

Problema central: el ONI es igual para todos los municipios en un mismo año.
Aunque el panel tenga ~100 000 filas, la variación que identifica el efecto
proviene de solo ~19 años. Un error estándar MCO ingenuo trata las filas como
independientes y es varias veces demasiado optimista. Por eso:

- Los intervalos se obtienen con **bootstrap por años** (se remuestrean años
  completos, preservando la correlación entre municipios del mismo año).
- La significancia se evalúa con una **prueba de permutación**: se baraja la
  serie del ONI entre años y se recalcula el efecto. Es exacta bajo la hipótesis
  nula de intercambiabilidad y no depende de supuestos asintóticos, que con 19
  conglomerados no se cumplen.
- La heterogeneidad cultivo x región se estima con **agrupamiento parcial**
  (Bayes empírico): las regiones con poca información se acercan a la media
  del cultivo en lugar de reportar efectos extremos por ruido.

Como ENSO no lo controla nadie en Colombia, es un *experimento natural*:
la variación año a año del ONI no está correlacionada con decisiones de los
productores, lo que hace creíble una lectura causal del efecto (condicionada
a la tendencia propia de cada serie).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config as C

OUTCOME = "anom_log_rend"  # anomalía de log-rendimiento con quiebre metodológico 2019


@dataclass
class Effect:
    grupo: str
    beta: float  # cambio en log-rendimiento por +1 °C de ONI
    efecto_pct: float  # misma cantidad como % del rendimiento
    ic_low_pct: float
    ic_high_pct: float
    se_ingenuo: float
    se_cluster_anio: float
    se_bootstrap: float
    p_permutacion: float
    n_obs: int
    n_series: int
    n_anios: int


# --------------------------------------------------------------------------
# Pendiente sobre agregados anuales (rápida para bootstrap y permutaciones)
# --------------------------------------------------------------------------
def _year_aggregates(df: pd.DataFrame, y: str, x: str, w: str | None) -> pd.DataFrame:
    d = df[[y, x, "anio"] + ([w] if w else [])].dropna()
    weights = d[w] if w else pd.Series(1.0, index=d.index)
    agg = (
        pd.DataFrame({"anio": d["anio"], "W": weights, "S": weights * d[y], "x": d[x]})
        .groupby("anio")
        .agg(W=("W", "sum"), S=("S", "sum"), x=("x", "first"))
    )
    return agg


def _slopes(K: np.ndarray, X: np.ndarray, S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pendiente MCO ponderada para cada fila de multiplicidades ``K`` (B x T).

    ``X`` puede ser (T,) o (B, T) cuando se permutan los valores del ONI.
    Equivale a ajustar y_i = a + b x_t sobre los microdatos de los años
    seleccionados, pero usando solo sumas por año.
    """
    KW = K * W
    xbar = (KW * X).sum(1) / KW.sum(1)
    abar = (K * S).sum(1) / KW.sum(1)
    dx = X - xbar[:, None]
    num = (K * dx * (S - W * abar[:, None])).sum(1)
    den = (KW * dx**2).sum(1)
    return num / den


def year_slope(df: pd.DataFrame, y: str, x: str, w: str | None = None) -> float:
    agg = _year_aggregates(df, y, x, w)
    return float(
        _slopes(np.ones((1, len(agg))), agg["x"].to_numpy(), agg["S"].to_numpy(), agg["W"].to_numpy())[0]
    )


def bootstrap_years(
    df: pd.DataFrame, y: str, x: str, w: str | None = None, n_boot: int = 4000, seed: int = C.RANDOM_SEED
) -> np.ndarray:
    agg = _year_aggregates(df, y, x, w)
    rng = np.random.default_rng(seed)
    T = len(agg)
    idx = rng.integers(0, T, size=(n_boot, T))
    K = np.zeros((n_boot, T))
    np.add.at(K, (np.repeat(np.arange(n_boot), T), idx.ravel()), 1)
    xs = agg["x"].to_numpy()
    ok = (K * (xs - xs.mean()) ** 2).sum(1) > 0  # descarta remuestras sin variación en x
    return _slopes(K[ok], xs, agg["S"].to_numpy(), agg["W"].to_numpy())


def permutation_pvalue(
    df: pd.DataFrame, y: str, x: str, w: str | None = None, n_perm: int = 4000, seed: int = C.RANDOM_SEED
) -> float:
    """p-valor bilateral: fracción de permutaciones del ONI con |efecto| >= observado."""
    agg = _year_aggregates(df, y, x, w)
    rng = np.random.default_rng(seed + 1)
    xs = agg["x"].to_numpy()
    S, W = agg["S"].to_numpy(), agg["W"].to_numpy()
    obs = _slopes(np.ones((1, len(xs))), xs, S, W)[0]
    Xp = np.array([rng.permutation(xs) for _ in range(n_perm)])
    null = _slopes(np.ones_like(Xp), Xp, S, W)
    return float((1 + (np.abs(null) >= abs(obs)).sum()) / (n_perm + 1))


def estimate_effect(
    df: pd.DataFrame,
    grupo: str,
    y: str = OUTCOME,
    x: str = "oni_anual",
    w: str | None = None,
    n_boot: int = 4000,
) -> Effect:
    d = df.dropna(subset=[y, x])
    beta = year_slope(d, y, x, w)
    boots = bootstrap_years(d, y, x, w, n_boot)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    model = smf.wls(f"{y} ~ {x}", data=d, weights=d[w] if w else np.ones(len(d)))
    naive = model.fit()
    clus = model.fit(cov_type="cluster", cov_kwds={"groups": d["anio"].to_numpy()})

    def pct(b: float) -> float:
        return float(100 * (np.exp(b) - 1))

    return Effect(
        grupo=grupo,
        beta=beta,
        efecto_pct=pct(beta),
        ic_low_pct=pct(lo),
        ic_high_pct=pct(hi),
        se_ingenuo=float(naive.bse[x]),
        se_cluster_anio=float(clus.bse[x]),
        se_bootstrap=float(np.std(boots, ddof=1)),
        p_permutacion=permutation_pvalue(d, y, x, w),
        n_obs=len(d),
        n_series=int(d["serie"].nunique()),
        n_anios=int(d["anio"].nunique()),
    )


# --------------------------------------------------------------------------
# Resultados principales
# --------------------------------------------------------------------------
def effects_by_crop(panel: pd.DataFrame, **kw) -> pd.DataFrame:
    rows = [asdict(estimate_effect(panel, "Todos los cultivos", **kw))]
    for crop, g in panel.groupby("cultivo"):
        rows.append(asdict(estimate_effect(g, C.CROPS[crop], **kw)))
    out = pd.DataFrame(rows)
    out["q_bh"] = np.nan
    crops = out["grupo"] != "Todos los cultivos"
    out.loc[crops, "q_bh"] = benjamini_hochberg(out.loc[crops, "p_permutacion"].to_numpy())
    out["significativo"] = out["p_permutacion"] < 0.05
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """q-valores de Benjamini-Hochberg: controlan la tasa de falsos descubrimientos.

    Con 12 cultivos, esperaríamos ~0.6 resultados con p < 0.05 por puro azar.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def phase_contrasts(panel: pd.DataFrame, n_boot: int = 4000, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Rendimiento medio en años Niño y Niña frente a años neutrales, por cultivo.

    El bootstrap remuestrea años *dentro* de cada fase para conservar el
    número de años Niño / Niña / neutrales observados.
    """
    rng = np.random.default_rng(seed)
    rows = []
    groups = [("Todos los cultivos", panel)] + [(C.CROPS[c], g) for c, g in panel.groupby("cultivo")]
    for name, g in groups:
        yr = g.groupby("anio").agg(S=(OUTCOME, "sum"), W=(OUTCOME, "size"), fase=("fase", "first"))
        by_phase = {f: yr[yr["fase"] == f] for f in ("Niño", "Neutral", "Niña")}
        if any(len(v) == 0 for v in by_phase.values()):
            continue

        def mean_of(sub: pd.DataFrame, idx: np.ndarray | None = None) -> float:
            s = sub.iloc[idx] if idx is not None else sub
            return s["S"].sum() / s["W"].sum()

        point = {f: mean_of(v) for f, v in by_phase.items()}
        boots = {f: [] for f in by_phase}
        for _ in range(n_boot):
            for f, v in by_phase.items():
                boots[f].append(mean_of(v, rng.integers(0, len(v), len(v))))
        for f in ("Niño", "Niña"):
            diff = np.array(boots[f]) - np.array(boots["Neutral"])
            d0 = point[f] - point["Neutral"]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append(
                {
                    "grupo": name,
                    "fase": f,
                    "diferencia_pct": 100 * (np.exp(d0) - 1),
                    "ic_low_pct": 100 * (np.exp(lo) - 1),
                    "ic_high_pct": 100 * (np.exp(hi) - 1),
                    "anios_fase": len(by_phase[f]),
                    "anios_neutral": len(by_phase["Neutral"]),
                }
            )
    return pd.DataFrame(rows)


def heterogeneity(
    panel: pd.DataFrame, min_obs: int = 300, min_series: int = 25, n_boot: int = 2000
) -> pd.DataFrame:
    """Efecto por cultivo x región con agrupamiento parcial hacia la media del cultivo."""
    rows = []
    for (crop, region), g in panel.groupby(["cultivo", "region"]):
        if len(g) < min_obs or g["serie"].nunique() < min_series:
            continue
        b = year_slope(g, OUTCOME, "oni_anual")
        se = float(np.std(bootstrap_years(g, OUTCOME, "oni_anual", n_boot=n_boot), ddof=1))
        rows.append({"cultivo": C.CROPS[crop], "region": region, "beta": b, "se": se, "n_obs": len(g)})
    het = pd.DataFrame(rows)
    het["beta_parcial"] = np.nan
    het["se_parcial"] = np.nan
    for _crop, g in het.groupby("cultivo"):
        shrunk, se_s = empirical_bayes(g["beta"].to_numpy(), g["se"].to_numpy())
        het.loc[g.index, "beta_parcial"] = shrunk
        het.loc[g.index, "se_parcial"] = se_s
    for col in ("beta", "beta_parcial"):
        het[col.replace("beta", "efecto_pct")] = 100 * (np.exp(het[col]) - 1)
    het["ic_low_pct"] = 100 * (np.exp(het["beta_parcial"] - 1.96 * het["se_parcial"]) - 1)
    het["ic_high_pct"] = 100 * (np.exp(het["beta_parcial"] + 1.96 * het["se_parcial"]) - 1)
    return het


def empirical_bayes(beta: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contracción de DerSimonian-Laird: estimaciones ruidosas se acercan a la media.

    Es la misma lógica que las pendientes aleatorias de un modelo mixto, pero
    usando errores estándar que ya respetan la correlación entre municipios
    del mismo año.
    """
    if len(beta) == 1:
        return beta.copy(), se.copy()
    w = 1 / se**2
    mu_fe = np.sum(w * beta) / w.sum()
    q = np.sum(w * (beta - mu_fe) ** 2)
    tau2 = max(0.0, (q - (len(beta) - 1)) / (w.sum() - np.sum(w**2) / w.sum()))
    w_re = 1 / (se**2 + tau2)
    mu = np.sum(w_re * beta) / w_re.sum()
    shrink = tau2 / (tau2 + se**2)
    post = mu + shrink * (beta - mu)
    post_se = np.sqrt(shrink * se**2 + (1 - shrink) ** 2 / w_re.sum())
    return post, post_se


def climate_channel(panel: pd.DataFrame) -> pd.DataFrame:
    """Efecto de la lluvia y la temperatura sobre el rendimiento, con efectos fijos de año.

    Con efectos fijos de año, el efecto se identifica comparando municipios en
    el *mismo* año: absorbe cualquier choque nacional (incluido el cambio
    metodológico de la EVA) y aprovecha que El Niño no seca todo el país por igual.
    El déficit y el exceso de lluvia entran por separado porque ambos dañan.
    Errores agrupados por departamento (la lluvia está correlacionada en el espacio).
    """
    d = panel.dropna(subset=[OUTCOME, "lluvia_anom_pct_anual", "temp_anom_anual"]).copy()
    d["deficit_10pct"] = np.clip(-d["lluvia_anom_pct_anual"], 0, None) / 10
    d["exceso_10pct"] = np.clip(d["lluvia_anom_pct_anual"], 0, None) / 10
    d["anio_fe"] = d["anio"].astype(str)  # efecto fijo de año (categórica)
    rows = []
    groups = [("Todos los cultivos", d)] + [(C.CROPS[c], g) for c, g in d.groupby("cultivo")]
    for name, g in groups:
        fit = smf.ols(f"{OUTCOME} ~ deficit_10pct + exceso_10pct + temp_anom_anual + anio_fe", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g["cod_dep"].astype(int).to_numpy()}
        )
        for var, label in (
            ("deficit_10pct", "Cada 10 % menos de lluvia"),
            ("exceso_10pct", "Cada 10 % más de lluvia"),
            ("temp_anom_anual", "Cada +1 °C de temperatura"),
        ):
            b, se = fit.params[var], fit.bse[var]
            rows.append(
                {
                    "grupo": name,
                    "variable": label,
                    "efecto_pct": 100 * (np.exp(b) - 1),
                    "ic_low_pct": 100 * (np.exp(b - 1.96 * se) - 1),
                    "ic_high_pct": 100 * (np.exp(b + 1.96 * se) - 1),
                    "p_valor": float(fit.pvalues[var]),
                    "n_obs": int(fit.nobs),
                }
            )
    return pd.DataFrame(rows)


def enso_rainfall_map(clima_mun: pd.DataFrame, enso_annual: pd.DataFrame) -> pd.DataFrame:
    """Sensibilidad de la lluvia de cada municipio al ONI (1991-2025, 35 años)."""
    d = clima_mun.merge(enso_annual[["anio", "oni_anual"]], on="anio").dropna(
        subset=["lluvia_anom_pct_anual", "oni_anual"]
    )
    d = d[d["anio"] <= C.YEAR_MAX]

    def _fit(g: pd.DataFrame) -> pd.Series:
        x, y = g["oni_anual"].to_numpy(), g["lluvia_anom_pct_anual"].to_numpy()
        b = np.polyfit(x, y, 1)[0]
        r = np.corrcoef(x, y)[0, 1]
        return pd.Series({"lluvia_pct_por_oni": b, "correlacion": r, "n_anios": len(g)})

    return d.groupby("cod_mun")[["oni_anual", "lluvia_anom_pct_anual"]].apply(_fit).reset_index()


def robustness(panel: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """El efecto nacional bajo especificaciones alternativas razonables."""
    specs = []

    def add(name: str, df: pd.DataFrame, **kw) -> None:
        e = estimate_effect(df, name, n_boot=n_boot, **kw)
        specs.append(asdict(e))

    add("Principal: tendencia por serie + quiebre 2019", panel)
    add("Sin quiebre metodológico 2019", panel, y="anom_log_rend_sin_quiebre")
    add("ONI julio(t-1)-junio(t)", panel, x="oni_jul_jun")
    add("Ponderado por área cosechada", panel, w="area_cosechada")
    long = panel[panel.groupby("serie")["anio"].transform("nunique") >= 15]
    add("Solo series con 15+ años", long)
    add("Excluyendo 2018 (cobertura parcial)", panel[panel["anio"] != 2018])
    rep_share = panel.groupby("serie")["repetido"].transform("mean")
    add("Solo series con < 25 % de datos repetidos", panel[rep_share < 0.25])
    add("Solo base 2007-2018", panel[panel["anio"] <= 2018])
    out = pd.DataFrame(specs).rename(columns={"grupo": "especificacion"})
    return out
