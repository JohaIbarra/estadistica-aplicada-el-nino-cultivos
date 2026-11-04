import numpy as np
import pandas as pd
import pytest

from agroclima import climate, enso

ONI_TEXT = """ 2020 2022
2020   0.5   0.5   0.4   0.2   -0.1  -0.3  -0.4  -0.6  -0.9  -1.2  -1.3  -1.2
2021  -1.0  -0.9  -0.8  -0.7  -0.5  -0.4  -0.4  -0.5  -0.7  -0.8  -1.0  -1.0
2022  -1.0  -0.9  -1.0  -1.1  -1.0  -0.9  -0.8  -0.9  -1.0  -99.9 -99.9 -99.9
  -99.9
 ONI from CPC
"""


def test_parse_oni_drops_missing_values():
    oni = enso.parse_oni(ONI_TEXT)
    assert len(oni) == 33
    assert oni["oni"].min() > -50
    assert oni.iloc[-1].tolist() == [2022, 9, -1.0]


def test_official_episodes_requires_five_consecutive_months():
    oni = pd.DataFrame({"anio": 2000, "mes": range(1, 13), "oni": [0.6] * 4 + [0] + [-0.6] * 5 + [0, 0]})
    phase = enso.official_episodes(oni)
    assert (phase.iloc[:4] == "Neutral").all()  # 4 meses no alcanzan
    assert (phase.iloc[5:10] == "Niña").all()


def test_annual_exposure_windows_use_only_allowed_months():
    oni = enso.parse_oni(ONI_TEXT)
    ann = enso.annual_exposure(oni).set_index("anio")
    assert ann.loc[2021, "oni_anual"] == pytest.approx(oni[oni.anio == 2021]["oni"].mean())
    # El pronóstico de 2022 solo puede ver julio-septiembre de 2021
    assert ann.loc[2022, "oni_jas_prev"] == pytest.approx(np.mean([-0.4, -0.5, -0.7]))
    assert ann.loc[2022, "oni_tend_prev"] == pytest.approx(-0.7 - (-0.4))
    assert ann.loc[2021, "fase"] == "Niña"
    # 2022 está incompleto: no hay ONI anual, pero sí hay insumos para pronosticar 2023
    assert np.isnan(ann.loc[2022, "oni_anual"])
    assert ann.loc[2023, "oni_jas_prev"] == pytest.approx(np.mean([-0.8, -0.9, -1.0]))


def _monthly(years, rain_fn):
    rows = []
    for y in years:
        for m in range(1, 13):
            rows.append(
                {
                    "cod_mun": "05001",
                    "anio": y,
                    "mes": m,
                    "lluvia_mm": rain_fn(y, m),
                    "temp_c": 20.0,
                    "tmax_c": 26.0,
                }
            )
    return pd.DataFrame(rows)


def test_climate_anomalies_against_1991_2020_normal():
    monthly = _monthly(range(1991, 2023), lambda y, m: 100.0 if y != 2021 else 50.0)
    feats = climate.annual_features(climate.add_anomalies(monthly)).set_index("anio")
    # La normal 1991-2020 es 100 mm por mes; 2021 tiene la mitad
    assert feats.loc[2021, "lluvia_anom_pct_anual"] == pytest.approx(-50)
    assert feats.loc[2021, "meses_secos_anual"] == 0  # 50 % no es "< 50 %"
    # El rasgo _prev de 2022 resume enero-septiembre de 2021
    assert feats.loc[2022, "lluvia_anom_pct_prev"] == pytest.approx(-50)


def test_climate_prev_ignores_october_to_december():
    base = _monthly(range(1991, 2023), lambda y, m: 100.0)
    altered = _monthly(range(1991, 2023), lambda y, m: 100.0 if not (y == 2021 and m >= 10) else 999.0)
    a = climate.annual_features(climate.add_anomalies(base)).set_index("anio")
    b = climate.annual_features(climate.add_anomalies(altered)).set_index("anio")
    assert a.loc[2022, "lluvia_anom_pct_prev"] == pytest.approx(b.loc[2022, "lluvia_anom_pct_prev"])
    assert a.loc[2021, "lluvia_anom_pct_anual"] != pytest.approx(b.loc[2021, "lluvia_anom_pct_anual"])
