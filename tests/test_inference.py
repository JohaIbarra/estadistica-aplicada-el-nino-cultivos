import numpy as np
import pandas as pd
import pytest

from agroclima import inference, panel


def test_detrend_removes_series_trend_and_2019_break():
    years = np.arange(2007, 2026)
    logy = 1 + 0.02 * (years - 2007) + 0.3 * (years >= 2019)
    df = pd.DataFrame({"serie": "a", "anio": years, "rendimiento": np.exp(logy)})
    with_break = panel.detrend_log_yield(df, break_2019=True)
    without = panel.detrend_log_yield(df, break_2019=False)
    assert np.allclose(with_break, 0, atol=1e-10)
    assert np.abs(without).max() > 0.05  # sin el quiebre queda un artefacto


def test_year_slope_matches_micro_ols(synthetic_panel):
    d = synthetic_panel.copy()
    d["y"] = np.log(d["rendimiento"])
    fast = inference.year_slope(d, "y", "oni_anual")
    slow = np.polyfit(d["oni_anual"], d["y"], 1)[0]
    assert fast == pytest.approx(slow, rel=1e-9)


def test_effect_recovers_known_value_and_is_honest_about_uncertainty(synthetic_panel):
    d = synthetic_panel.copy()
    d["anom_log_rend"] = panel.detrend_log_yield(d)
    eff = inference.estimate_effect(d, "test", n_boot=1000)
    assert eff.beta == pytest.approx(-0.08, abs=0.03)
    assert eff.p_permutacion < 0.05
    # Con un solo valor de ONI por año, el error ingenuo subestima la incertidumbre
    assert eff.se_bootstrap > 2 * eff.se_ingenuo


def test_permutation_pvalue_is_not_significant_without_effect(synthetic_panel, rng):
    d = synthetic_panel.copy()
    d["y"] = rng.normal(0, 0.1, len(d))
    assert inference.permutation_pvalue(d, "y", "oni_anual", n_perm=1000) > 0.05


def test_empirical_bayes_shrinks_noisy_estimates_toward_mean():
    beta = np.array([-0.10, -0.08, 0.40])
    se = np.array([0.02, 0.02, 0.30])
    post, post_se = inference.empirical_bayes(beta, se)
    assert abs(post[2] - post[:2].mean()) < abs(beta[2] - beta[:2].mean())
    assert np.all(post_se <= se + 1e-12)


def test_phase_contrasts_produces_both_phases(synthetic_panel):
    d = synthetic_panel.copy()
    d["anom_log_rend"] = panel.detrend_log_yield(d)
    out = inference.phase_contrasts(d, n_boot=200)
    sub = out[out["grupo"] == "Todos los cultivos"].set_index("fase")
    assert set(sub.index) == {"Niño", "Niña"}
    # Efecto negativo del ONI: años Niño peores que años Niña
    assert sub.loc["Niño", "diferencia_pct"] < sub.loc["Niña", "diferencia_pct"]


def test_benjamini_hochberg_known_values():
    q = inference.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))
    assert q.tolist() == pytest.approx([0.04, 0.05333333, 0.05333333, 0.20])


def test_flag_repeated_only_consecutive_years():
    df = pd.DataFrame(
        {"serie": ["a"] * 4, "anio": [2010, 2011, 2013, 2014], "rendimiento": [2.0, 2.0, 2.0, 2.5]}
    )
    assert panel.flag_repeated(df).tolist() == [False, True, False, False]
