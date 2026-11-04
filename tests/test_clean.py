import numpy as np
import pandas as pd
import pytest

from agroclima import clean


def test_normalize_text_removes_accents_and_spaces():
    s = pd.Series(["  Caña   panelera ", "Maíz", None])
    out = clean.normalize_text(s)
    assert out.iloc[0] == "CANA PANELERA"
    assert out.iloc[1] == "MAIZ"
    assert pd.isna(out.iloc[2])


@pytest.mark.parametrize(
    ("cultivo", "variante", "esperado"),
    [
        ("MAIZ", "MAIZ AMARILLO TRADICIONAL", ("MAIZ", "MAIZ TRADICIONAL")),
        ("MAIZ", "MAIZ TRADICIONAL", ("MAIZ", "MAIZ TRADICIONAL")),
        ("MAIZ", "MAIZ BLANCO TECNIFICADO", ("MAIZ", "MAIZ TECNIFICADO")),
        ("MAIZ", "MAIZ FORRAJERO", (None, None)),
        ("CANA", "CANA PANELERA", ("CANA PANELERA", "CANA PANELERA")),
        ("CANA PANELERA", "CANA PANELERA", ("CANA PANELERA", "CANA PANELERA")),
        ("CANA", "CANA DE AZUCAR", (None, None)),
        ("PAPA", "PAPA CRIOLLA", ("PAPA", "PAPA CRIOLLA")),
        ("PAPA", "PAPA TODAS LAS VARIEDADES", ("PAPA", "PAPA")),
        ("PAPAYA", "PAPAYA", (None, None)),
        ("PLATANO", "PLATANO CONSUMO INTERNO", ("PLATANO", "PLATANO")),
        ("ARROZ", "ARROZ SECANO MECANIZADO", ("ARROZ", "ARROZ SECANO MECANIZADO")),
        ("CAFE", "CAFE", ("CAFE", "CAFE")),
    ],
)
def test_harmonize_crop_is_comparable_across_bases(cultivo, variante, esperado):
    assert clean.harmonize_crop(cultivo, variante) == esperado


def test_repair_coordinate_recovers_lost_decimal_point():
    assert clean.repair_coordinate(525.0, -4.3, 13.6) == pytest.approx(5.25)
    assert clean.repair_coordinate(-74916666.875, -82.0, -66.8) == pytest.approx(-74.916666875)
    assert clean.repair_coordinate(4.5, -4.3, 13.6) == 4.5
    assert np.isnan(clean.repair_coordinate(np.nan, -4.3, 13.6))


def _raw(**over):
    base = {
        "cod_dep": "5",
        "cod_mun": "5001",
        "municipio": "Medellín",
        "departamento": "Antioquia",
        "cultivo_raw": "Café",
        "variante_raw": "Café",
        "anio": "2010",
        "periodo": "2010",
        "area_sembrada": "100",
        "area_cosechada": "90",
        "produccion": "90",
        "fuente": "test",
    }
    base.update(over)
    return base


def test_clean_eva_pads_codes_and_drops_impossible_records():
    raw = pd.DataFrame(
        [
            _raw(),
            _raw(produccion="0"),  # sin cosecha
            _raw(area_cosechada="200"),  # cosechada > sembrada
            _raw(cultivo_raw="Papaya", variante_raw="Papaya"),  # fuera de alcance
            _raw(anio="2006"),  # fuera de rango
        ]
    )
    df, log = clean.clean_eva(raw)
    assert len(df) == 1
    assert df["cod_mun"].iloc[0] == "05001"
    assert df["cod_dep"].iloc[0] == "05"
    assert log["filas"].iloc[0] == 5
    assert log["filas"].iloc[1:].sum() == 4


def test_to_annual_weights_semesters_by_area():
    raw = pd.DataFrame(
        [
            _raw(
                cultivo_raw="Maíz",
                variante_raw="Maíz tradicional",
                periodo="2010A",
                area_cosechada="10",
                produccion="10",
            ),
            _raw(
                cultivo_raw="Maíz",
                variante_raw="Maíz amarillo tradicional",
                periodo="2010B",
                area_cosechada="30",
                produccion="90",
            ),
        ]
    )
    df, _ = clean.clean_eva(raw)
    annual = clean.to_annual(df)
    assert len(annual) == 1
    # (10 + 90) / (10 + 30) = 2.5 t/ha, no el promedio simple (1 + 3) / 2 = 2
    assert annual["rendimiento"].iloc[0] == pytest.approx(2.5)
    assert annual["region"].iloc[0] == "Andina"


def test_flag_outliers_catches_typing_errors():
    rend = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 110.0, 1.02]
    df = pd.DataFrame({"serie": "a", "rendimiento": rend})
    flags = clean.flag_outliers(df)
    assert flags.tolist() == [False] * 6 + [True, False]


def test_flag_outliers_keeps_genuine_bad_harvests():
    # Serie muy estable con una caída real de 35 %: no es un error de digitación
    rend = [2.0, 2.02, 1.98, 2.0, 2.01, 1.3, 2.0, 1.99]
    df = pd.DataFrame({"serie": "a", "rendimiento": rend})
    assert not clean.flag_outliers(df).any()
