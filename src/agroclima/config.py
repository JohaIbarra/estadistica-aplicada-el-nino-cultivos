"""Configuración central: rutas, fuentes de datos y decisiones de análisis.

Todas las decisiones que afectan los resultados (umbrales, ventanas, cultivos)
viven aquí para que sean explícitas, auditables y fáciles de variar en
análisis de sensibilidad.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("AGROCLIMA_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# --------------------------------------------------------------------------
# Fuentes
# --------------------------------------------------------------------------
SOCRATA = "https://www.datos.gov.co/resource"
EVA_HISTORICA_ID = "2pnw-mmge"  # EVA 2007-2018 (MinAgricultura)
EVA_RECIENTE_ID = "uejq-wxrr"  # EVA 2019-2025 (MinAgricultura / UPRA)
DIVIPOLA_ID = "pqwj-3fi4"  # Municipios con centroide (MinSalud)
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
# CHIRPS v2.0 (UCSB/CHC) vía IRI Data Library: lluvia mensual a 0.05° (~5 km),
# combina satélite y estaciones. Es la referencia para el trópico.
CHIRPS_URL = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation"
CHIRPS_BOX = (-79.2, -66.7, -4.4, 13.7)  # lon_min, lon_max, lat_min, lat_max
CHIRPS_RADIUS_DEG = 0.1  # ventana alrededor del centroide municipal (~5x5 píxeles)
# NASA POWER (MERRA-2) se usa solo para temperatura: su lluvia reanalizada no
# reproduce la variabilidad interanual en Colombia (ver reports/informe_tecnico.md).
POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
POWER_PARAMS = ("T2M", "T2M_MAX")
POWER_START, POWER_END = 1991, 2026  # 2026 parcial (hasta septiembre)

# Malla de MERRA-2 usada por NASA POWER (0.5° lat x 0.625° lon). Municipios que
# caen en la misma celda reciben el mismo clima, así que deduplicamos llamadas.
POWER_GRID_LAT, POWER_GRID_LON = 0.5, 0.625

# --------------------------------------------------------------------------
# Decisiones de análisis
# --------------------------------------------------------------------------
YEAR_MIN, YEAR_MAX = 2007, 2025
CLIMATOLOGY = (1991, 2020)  # normal climatológica OMM para anomalías

# Umbral ONI de NOAA: |ONI| >= 0.5 define episodio cálido (Niño) o frío (Niña).
ONI_THRESHOLD = 0.5

# Cultivos analizados: los de mayor área cosechada y relevancia para la
# seguridad alimentaria y exportaciones. Clave = nombre canónico.
CROPS: dict[str, str] = {
    "CAFE": "Café",
    "ARROZ": "Arroz",
    "MAIZ": "Maíz",
    "PAPA": "Papa",
    "PLATANO": "Plátano",
    "YUCA": "Yuca",
    "CANA PANELERA": "Caña panelera",
    "FRIJOL": "Fríjol",
    "CACAO": "Cacao",
    "PALMA DE ACEITE": "Palma de aceite",
    "BANANO": "Banano",
    "AGUACATE": "Aguacate",
}

# Filtros de calidad del panel
MIN_AREA_HA = 10.0  # descarta municipio-cultivo-año con área cosechada mínima
MIN_YEARS = 8  # años con dato para que una serie municipio-cultivo entre al panel
OUTLIER_MAD = 5.0  # anomalías a más de 5 MAD se consideran error de registro

# Pronóstico: "mala cosecha" = rendimiento del año t al menos 20 % por debajo
# de la mediana de los últimos 5 años *publicados* (t-2 ... t-6) de la misma serie.
BAD_HARVEST_DROP = 0.20
BASELINE_YEARS = 5
FIRST_TEST_YEAR = 2015
# Fecha de emisión del pronóstico: 1 de octubre del año t-1. Solo se usa
# información publicada antes de esa fecha (ONI hasta septiembre, clima hasta
# septiembre y rendimientos EVA hasta t-2, porque la EVA se publica con ~1 año
# de rezago).
FORECAST_LAST_CLIMATE_MONTH = 9
EVA_PUBLICATION_LAG = 2
FORECAST_YEAR = 2027


@dataclass(frozen=True)
class Region:
    nombre: str
    departamentos: tuple[str, ...]


# Regiones naturales (agrupación usada por IDEAM / DANE), por código DANE.
REGIONES: tuple[Region, ...] = (
    Region("Caribe", ("08", "13", "20", "23", "44", "47", "70", "88")),
    Region("Andina", ("05", "11", "15", "17", "25", "41", "54", "63", "66", "68", "73")),
    Region("Pacífica", ("19", "27", "52", "76")),
    Region("Orinoquía", ("50", "81", "85", "99")),
    Region("Amazonía", ("18", "86", "91", "94", "95", "97")),
)

DEPTO_A_REGION: dict[str, str] = {d: r.nombre for r in REGIONES for d in r.departamentos}

RANDOM_SEED = 20261104
