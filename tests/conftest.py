"""Datos sintéticos pequeños: los tests no dependen de la red ni de data/raw."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agroclima import config as C


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_panel(rng: np.random.Generator) -> pd.DataFrame:
    """Panel con efecto ENSO conocido: -8 % de rendimiento por +1 °C de ONI.

    Incluye un choque anual común a todos los municipios (como el clima real),
    que es lo que hace que los errores ingenuos sean demasiado optimistas.
    """
    years = np.arange(C.YEAR_MIN, C.YEAR_MAX + 1)
    oni = rng.normal(0, 0.8, len(years))
    year_shock = rng.normal(0, 0.04, len(years))
    rows = []
    for s in range(60):
        cod_mun = f"{5000 + s:05d}"
        crop = ["CAFE", "ARROZ", "MAIZ"][s % 3]
        level = rng.normal(1.0, 0.3)
        trend = rng.normal(0.01, 0.005)
        for t, o, shock in zip(years, oni, year_shock, strict=True):
            logy = level + trend * (t - years[0]) - 0.08 * o + shock + rng.normal(0, 0.1)
            rows.append(
                {
                    "cod_mun": cod_mun,
                    "cod_dep": "05",
                    "cultivo": crop,
                    "variante": crop,
                    "region": "Andina",
                    "serie": f"{cod_mun}|{crop}",
                    "anio": int(t),
                    "rendimiento": float(np.exp(logy)),
                    "area_cosechada": 100.0,
                    "oni_anual": o,
                    "fase": "Niño" if o >= 0.5 else ("Niña" if o <= -0.5 else "Neutral"),
                    "lat": 6.0,
                    "lon": -75.0,
                    "elevacion_m": 1500.0,
                }
            )
    return pd.DataFrame(rows)
