"""Dashboard: El Niño, La Niña y los cultivos de Colombia.

Ejecutar con:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permite usar el paquete `agroclima` aunque no esté instalado (p. ej. en un despliegue mínimo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

st.set_page_config(
    page_title="El Niño y los cultivos de Colombia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .nota {opacity: 0.8; font-size: 0.92rem;}
      /* Fondo translúcido y color heredado: legible en tema claro y oscuro */
      .hallazgo {border-left: 3px solid #2a78d6; padding: 0.5rem 0.9rem; margin: 0.4rem 0 0.8rem 0;
                 background: rgba(42, 120, 214, 0.08); border-radius: 0 6px 6px 0; color: inherit;}
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("views/resumen.py", title="Resumen", icon=":material/insights:", default=True),
    st.Page("views/alerta.py", title="Alerta temprana 2027", icon=":material/warning:"),
    st.Page("views/efecto.py", title="Efecto de El Niño", icon=":material/thermostat:"),
    st.Page("views/explorador.py", title="Explorador de datos", icon=":material/map:"),
    st.Page("views/metodologia.py", title="Metodología y validación", icon=":material/science:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.caption(
        "Datos públicos: EVA (MinAgricultura), CHIRPS (UCSB), NASA POWER, ONI (NOAA). "
        "Código y metodología en el repositorio del proyecto."
    )

nav.run()
