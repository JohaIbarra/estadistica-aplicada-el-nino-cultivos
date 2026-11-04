"""El test más importante del proyecto: el pronóstico no ve el futuro."""

import numpy as np
import pandas as pd
import pytest

from agroclima import config as C
from agroclima import predict


def test_history_features_do_not_use_years_after_t_minus_2(synthetic_panel):
    t = 2020
    base = predict.history_features(synthetic_panel, [t]).set_index("serie").sort_index()

    tampered = synthetic_panel.copy()
    future = tampered["anio"] >= t - 1  # t-1 aún no está publicado al emitir el pronóstico
    tampered.loc[future, "rendimiento"] *= 50
    tampered.loc[future, "area_cosechada"] *= 50
    after = predict.history_features(tampered, [t]).set_index("serie").sort_index()

    pd.testing.assert_frame_equal(base, after)


def test_history_features_do_change_when_known_past_changes(synthetic_panel):
    t = 2020
    base = predict.history_features(synthetic_panel, [t])
    tampered = synthetic_panel.copy()
    tampered.loc[tampered["anio"] == t - 2, "rendimiento"] *= 3
    after = predict.history_features(tampered, [t])
    assert not np.allclose(base["desvio_ultimo"], after["desvio_ultimo"])


def test_label_definition(synthetic_panel):
    years = list(range(2013, 2026))
    enso = pd.DataFrame({"anio": years, "oni_jas_prev": 0.0, "oni_tend_prev": 0.0})
    clima = pd.DataFrame(columns=["cod_mun", "anio", "lluvia_anom_pct_prev"])
    clima = clima.astype({"cod_mun": str, "anio": int, "lluvia_anom_pct_prev": float})
    ex = predict.build_examples(synthetic_panel, enso, clima, years)
    row = ex.iloc[0]
    expected = np.log(row["rendimiento"]) < row["base_log"] + np.log(1 - C.BAD_HARVEST_DROP)
    assert row["mala_cosecha"] == float(expected)
    assert ex["mala_cosecha"].isin([0.0, 1.0]).all()


def test_backtest_trains_only_on_published_years(synthetic_panel, monkeypatch):
    years = list(range(2013, 2026))
    enso = pd.DataFrame({"anio": years, "oni_jas_prev": 0.1, "oni_tend_prev": 0.0})
    clima = pd.DataFrame({"cod_mun": pd.Series(dtype=str), "anio": pd.Series(dtype=int)})
    ex = predict.build_examples(synthetic_panel, enso, clima, years)
    ex["mala_cosecha"] = (np.arange(len(ex)) % 4 == 0).astype(float)  # garantiza ambas clases

    seen: dict[int, int] = {}
    real_fit = predict.fit_lgbm

    def spy(train, seed=C.RANDOM_SEED):
        seen[len(seen)] = int(train["anio"].max())
        return real_fit(train, seed)

    monkeypatch.setattr(predict, "fit_lgbm", spy)
    metrics, _ = predict.backtest(ex, first_test=2018)
    tested = sorted(metrics["anio"].unique())
    assert [seen[i] for i in range(len(tested))] == [T - C.EVA_PUBLICATION_LAG for T in tested]


@pytest.mark.parametrize("n", [50])
def test_metrics_top_decile(n):
    y = np.array([1] * 5 + [0] * (n - 5))
    p = np.linspace(1, 0, n)
    m = predict._metrics(y, p)
    assert m["precision_top10"] == 1.0
    assert m["recall_top10"] == 1.0


def test_calibration_uses_only_past_backtest_years():
    rng = np.random.default_rng(1)
    rows = []
    for year in range(2017, 2024):
        p = rng.uniform(0, 1, 300)
        rows.append(
            pd.DataFrame(
                {"anio": year, "p_lgbm": p, "mala_cosecha": (rng.uniform(0, 1, 300) < p**2).astype(float)}
            )
        )
    preds = pd.concat(rows, ignore_index=True)
    cal = predict.calibrate_backtest(preds)
    # Los primeros años no tienen pasado suficiente: quedan crudos
    first = preds["anio"] == 2017
    assert np.allclose(cal[first], preds.loc[first, "p_lgbm"])
    # Cambiar el futuro no altera la calibración de 2020
    tampered = preds.copy()
    tampered.loc[tampered["anio"] >= 2019, "mala_cosecha"] = 1.0
    cal2 = predict.calibrate_backtest(tampered)
    y2020 = preds["anio"] == 2020
    assert np.allclose(cal[y2020], cal2[y2020])


def test_persistence_baseline_maps_last_outcome_to_train_rates():
    train = pd.DataFrame(
        {"mala_ultima": [1.0, 1.0, 0.0, 0.0, np.nan], "mala_cosecha": [1.0, 0.0, 0.0, 0.0, 1.0]}
    )
    test = pd.DataFrame({"mala_ultima": [1.0, 0.0, np.nan]})
    assert predict.persistence(train, test).tolist() == [0.5, 0.0, 1.0]


def test_metrics_never_report_accuracy():
    m = predict._metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert "accuracy" not in m
    assert {"pr_auc", "precision_top10", "precision_top20", "tasa_base"} <= set(m)
