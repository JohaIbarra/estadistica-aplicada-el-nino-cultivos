import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agroclima import config as C
from agroclima import viz
from charts import forest
from data import load
from fmt import num, pct, share

eff = load("efectos_cultivo")
fase = load("efectos_fase")
het = load("efectos_region")
canal = load("canal_climatico")
rob = load("robustez")
rain = load("lluvia_vs_oni").merge(load("municipios"), on="cod_mun")

st.title("¿Cuánto afecta El Niño al rendimiento?")
st.markdown(
    "<p class='nota'>El Niño no lo controla nadie en Colombia, así que funciona como un <b>experimento natural</b>: "
    "comparamos el rendimiento de cada municipio-cultivo en años con distinto ONI, después de quitar su "
    "tendencia propia. Los intervalos se calculan remuestreando <b>años completos</b> (el ONI solo varía entre "
    "años) y la significancia con una <b>prueba de permutación</b> del ONI.</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Por cultivo", "Por región", "Mecanismo: lluvia y calor", "El Niño y la lluvia", "Robustez"]
)

with tab1:
    mode = st.radio("Medida", ["Por cada +1 °C de ONI", "Años Niño / Niña vs. neutrales"], horizontal=True)
    if mode.startswith("Por cada"):
        d = eff.sort_values("efecto_pct")
        d = d.assign(etiqueta=d["grupo"])
        st.plotly_chart(
            forest(
                d,
                "etiqueta",
                "efecto_pct",
                "ic_low_pct",
                "ic_high_pct",
                "Cambio en el rendimiento por +1 °C de ONI",
            ),
            use_container_width=True,
        )
        show = eff[
            ["grupo", "efecto_pct", "ic_low_pct", "ic_high_pct", "p_permutacion", "q_bh", "n_obs", "n_series"]
        ]
        st.dataframe(
            show.rename(
                columns={
                    "grupo": "Cultivo",
                    "efecto_pct": "Efecto %",
                    "ic_low_pct": "IC inf. %",
                    "ic_high_pct": "IC sup. %",
                    "p_permutacion": "p (permutación)",
                    "q_bh": "q (B-H)",
                    "n_obs": "Obs.",
                    "n_series": "Series",
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.2f")
                for c in ["Efecto %", "IC inf. %", "IC sup. %", "p (permutación)", "q (B-H)"]
            },
        )
        st.caption(
            "q (B-H): p-valor corregido por comparaciones múltiples (Benjamini-Hochberg). Con 12 cultivos, alguno "
            "saldría con p < 0,05 por azar; el q controla esa tasa de falsos descubrimientos."
        )
    else:
        f = fase[fase["fase"] == st.radio("Fase", ["Niño", "Niña"], horizontal=True)].sort_values(
            "diferencia_pct"
        )
        st.plotly_chart(
            forest(
                f,
                "grupo",
                "diferencia_pct",
                "ic_low_pct",
                "ic_high_pct",
                "Diferencia frente a años neutrales",
            ),
            use_container_width=True,
        )
        st.caption(
            f"Solo hay {int(f['anios_fase'].iloc[0])} años de esta fase en 2007-2025, por eso los intervalos son anchos. "
            "La medida continua (por °C de ONI) usa todos los años y es más precisa."
        )

with tab2:
    regions = [r.nombre for r in C.REGIONES]
    piv = het.pivot_table(index="cultivo", columns="region", values="efecto_pct_parcial").reindex(
        columns=regions
    )
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    lo = het.pivot_table(index="cultivo", columns="region", values="ic_low_pct").reindex(
        index=piv.index, columns=regions
    )
    hi = het.pivot_table(index="cultivo", columns="region", values="ic_high_pct").reindex(
        index=piv.index, columns=regions
    )
    lim = float(np.nanmax(np.abs(piv.to_numpy())))
    fig = go.Figure(
        go.Heatmap(
            z=piv.to_numpy(),
            x=regions,
            y=piv.index,
            zmid=0,
            zmin=-lim,
            zmax=lim,
            colorscale=[[i / (len(viz.DIVERGING) - 1), c] for i, c in enumerate(viz.DIVERGING)],
            customdata=np.dstack([lo.to_numpy(), hi.to_numpy()]),
            text=np.where(
                np.isfinite(piv.to_numpy()), np.char.mod("%+.1f", np.nan_to_num(piv.to_numpy())), ""
            ),
            texttemplate="%{text}",
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y} · %{x}</b><br>%{z:+.1f} % por °C<br>IC 95 %: %{customdata[0]:+.1f} a %{customdata[1]:+.1f}<extra></extra>",
            colorbar=dict(title="% por °C", thickness=12),
        )
    )
    fig.update_layout(**viz.plotly_layout(height=520))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Agrupamiento parcial (Bayes empírico): cada región se acerca al promedio de su cultivo en proporción a su "
        "ruido, igual que las pendientes aleatorias de un modelo mixto. Celdas vacías: menos de 300 observaciones."
    )

with tab3:
    crop = st.selectbox("Cultivo", ["Todos los cultivos", *sorted(C.CROPS.values())])
    d = canal[canal["grupo"] == crop]
    st.plotly_chart(
        forest(
            d, "variable", "efecto_pct", "ic_low_pct", "ic_high_pct", "Cambio en el rendimiento", height=220
        ),
        use_container_width=True,
    )
    st.markdown(
        "Con **efectos fijos de año** se comparan municipios dentro del *mismo* año: se elimina cualquier choque "
        "nacional (incluido el cambio de metodología de la EVA) y el efecto se identifica porque El Niño no seca ni "
        "calienta a todo el país por igual. Errores agrupados por departamento."
    )

with tab4:
    lim = float(np.nanpercentile(np.abs(rain["lluvia_pct_por_oni"]), 98))
    fig = px.scatter_map(
        rain,
        lat="lat",
        lon="lon",
        color="lluvia_pct_por_oni",
        hover_name="municipio",
        hover_data={
            "departamento": True,
            "lluvia_pct_por_oni": ":+.1f",
            "correlacion": ":.2f",
            "lat": False,
            "lon": False,
        },
        labels={
            "lluvia_pct_por_oni": "% lluvia por °C",
            "correlacion": "Correlación",
            "departamento": "Depto.",
        },
        color_continuous_scale=viz.DIVERGING,
        range_color=(-lim, lim),
        color_continuous_midpoint=0,
        zoom=4.3,
        center={"lat": 5.5, "lon": -74.0},
        map_style="carto-positron",
        height=600,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(**viz.plotly_layout(margin=dict(l=0, r=0, t=10, b=0)))
    st.plotly_chart(fig, use_container_width=True)
    frac_seco = (rain["lluvia_pct_por_oni"] < 0).mean()
    by_region = rain.groupby("region")["lluvia_pct_por_oni"].median().sort_values()
    detail = ", ".join(f"{r} {pct(v, 0).replace(' %', '')} %" for r, v in by_region.items())
    st.markdown(
        f"En **{share(frac_seco, 0)}** de los municipios la lluvia anual disminuye cuando sube el ONI (CHIRPS 1991-2025). "
        f"Cambio mediano por cada +1 °C de ONI, por región: {detail}."
    )

with tab5:
    st.dataframe(
        rob[["especificacion", "efecto_pct", "ic_low_pct", "ic_high_pct", "p_permutacion", "n_obs"]].rename(
            columns={
                "especificacion": "Especificación",
                "efecto_pct": "Efecto %",
                "ic_low_pct": "IC inf. %",
                "ic_high_pct": "IC sup. %",
                "p_permutacion": "p (permutación)",
                "n_obs": "Obs.",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in ["Efecto %", "IC inf. %", "IC sup. %", "p (permutación)"]
        },
    )
    n = eff.iloc[0]
    st.markdown(
        f"""
**Por qué no basta el error estándar por defecto.** Para el efecto nacional, el error MCO ingenuo es
{num(n["se_ingenuo"], 4)}; el bootstrap por años da {num(n["se_bootstrap"], 4)}, **{num(n["se_bootstrap"] / n["se_ingenuo"], 1)} veces mayor**.
Con el error ingenuo, cualquier diferencia pequeña parecería "significativa". La especificación sin quiebre
metodológico muestra otro sesgo: el salto de la EVA en 2019 se confundiría con el clima de ese año.
"""
    )
