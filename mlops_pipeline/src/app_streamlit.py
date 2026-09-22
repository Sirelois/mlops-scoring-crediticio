"""
app_streamlit.py
Dashboard de monitoreo de Data Drift para el modelo de Scoring Crediticio (PI M5).

Para correr: streamlit run app_streamlit.py (parado en mlops_pipeline/src/)

Secciones:
1. Visualización de métricas — comparación histórico vs. actual, tabla de
   métricas por variable con semáforo de alerta.
2. Análisis temporal — evolución del PSI a lo largo del tiempo.
3. Recomendaciones — mensajes automáticos según severidad del drift.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model_monitoring import (
    cargar_datos_para_monitoreo,
    dividir_historico_actual,
    generar_reporte_drift,
    drift_evolucion_temporal,
    generar_recomendaciones,
    UMBRAL_PSI_ALERTA,
    UMBRAL_PSI_CRITICO,
)

st.set_page_config(page_title="Monitoreo de Data Drift — Scoring Crediticio", layout="wide")


@st.cache_data
def cargar_todo():
    ruta_csv = Path(__file__).resolve().parent.parent.parent / "Base_de_datos.csv"
    df = cargar_datos_para_monitoreo(str(ruta_csv))
    historico, actual, fecha_corte = dividir_historico_actual(df)
    reporte = generar_reporte_drift(historico, actual)
    return df, historico, actual, fecha_corte, reporte


df, historico, actual, fecha_corte, reporte = cargar_todo()

st.title("📊 Monitoreo de Data Drift — Scoring Crediticio")
st.caption(
    f"Población histórica (referencia): {len(historico):,} créditos hasta "
    f"{fecha_corte.date()} | Población actual (a monitorear): {len(actual):,} "
    f"créditos desde esa fecha"
)

tab1, tab2, tab3 = st.tabs(
    ["📈 Visualización de métricas", "🕒 Análisis temporal", "💡 Recomendaciones"]
)

# ------------------------------------------------------------------
# TAB 1: Visualización de métricas
# ------------------------------------------------------------------
with tab1:
    st.subheader("Tabla de métricas de drift por variable")

    def semaforo(row):
        if row["critico"]:
            return "🔴 Crítico"
        elif row["alerta"]:
            return "🟡 Alerta"
        else:
            return "🟢 Estable"

    reporte_vista = reporte.copy()
    reporte_vista["estado"] = reporte_vista.apply(semaforo, axis=1)
    columnas_mostrar = ["variable", "tipo", "estado", "ks_pvalue", "psi", "jensen_shannon", "chi2_pvalue"]
    st.dataframe(reporte_vista[columnas_mostrar], use_container_width=True, hide_index=True)

    st.markdown(
        f"**Umbrales de referencia:** PSI > {UMBRAL_PSI_ALERTA} = alerta moderada, "
        f"PSI > {UMBRAL_PSI_CRITICO} = crítico. KS test: p-value < 0.05 indica "
        f"distribuciones significativamente distintas."
    )

    st.divider()
    st.subheader("Comparación de distribuciones: histórico vs. actual")

    variables_numericas = reporte[reporte["tipo"] == "numerica"]["variable"].tolist()
    variable_elegida = st.selectbox("Elegí una variable numérica para visualizar", variables_numericas)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(historico[variable_elegida].dropna(), bins=30, alpha=0.6, label="Histórico", color="steelblue")
        ax.hist(actual[variable_elegida].dropna(), bins=30, alpha=0.6, label="Actual", color="orange")
        ax.set_title(f"Distribución: {variable_elegida}")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fila = reporte[reporte["variable"] == variable_elegida].iloc[0]
        st.metric("PSI", f"{fila['psi']:.4f}")
        st.metric("KS p-value", f"{fila['ks_pvalue']:.4f}")
        st.metric("Jensen-Shannon", f"{fila['jensen_shannon']:.4f}")
        estado = "🔴 Crítico" if fila["critico"] else ("🟡 Alerta" if fila["alerta"] else "🟢 Estable")
        st.markdown(f"**Estado:** {estado}")

    st.divider()
    st.subheader("Variables categóricas")
    variables_categoricas = reporte[reporte["tipo"] == "categorica"]["variable"].tolist()
    var_cat_elegida = st.selectbox("Elegí una variable categórica", variables_categoricas)

    col3, col4 = st.columns(2)
    with col3:
        dist_hist = historico[var_cat_elegida].value_counts(normalize=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        dist_hist.plot(kind="bar", ax=ax2, color="steelblue")
        ax2.set_title(f"Histórico: {var_cat_elegida}")
        st.pyplot(fig2)
    with col4:
        dist_act = actual[var_cat_elegida].value_counts(normalize=True)
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        dist_act.plot(kind="bar", ax=ax3, color="orange")
        ax3.set_title(f"Actual: {var_cat_elegida}")
        st.pyplot(fig3)


# ------------------------------------------------------------------
# TAB 2: Análisis temporal
# ------------------------------------------------------------------
with tab2:
    st.subheader("Evolución del drift a lo largo del tiempo")
    st.caption(
        "Se divide el dataset en 6 ventanas cronológicas y se calcula el PSI "
        "de cada ventana contra la primera (referencia fija), para detectar "
        "tendencias o cambios abruptos."
    )

    variable_temporal = st.selectbox(
        "Variable a analizar en el tiempo",
        reporte[reporte["tipo"] == "numerica"]["variable"].tolist(),
        key="temporal",
    )

    evolucion = drift_evolucion_temporal(df, variable_temporal, n_periodos=6)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(evolucion["periodo"], evolucion["psi"], marker="o", color="steelblue")
    ax4.axhline(UMBRAL_PSI_ALERTA, color="orange", linestyle="--", label=f"Umbral alerta ({UMBRAL_PSI_ALERTA})")
    ax4.axhline(UMBRAL_PSI_CRITICO, color="red", linestyle="--", label=f"Umbral crítico ({UMBRAL_PSI_CRITICO})")
    ax4.set_xlabel("Período (0 = referencia)")
    ax4.set_ylabel("PSI acumulado vs. período 0")
    ax4.set_title(f"Evolución del PSI — {variable_temporal}")
    ax4.legend()
    st.pyplot(fig4)

    st.dataframe(evolucion, use_container_width=True, hide_index=True)

    # Detección simple de cambio abrupto: salto grande entre períodos consecutivos
    evolucion["salto"] = evolucion["psi"].diff().abs()
    salto_max = evolucion["salto"].max()
    if salto_max > UMBRAL_PSI_ALERTA:
        periodo_salto = evolucion.loc[evolucion["salto"].idxmax(), "periodo"]
        st.warning(
            f"⚠️ Se detectó un cambio abrupto en el período {int(periodo_salto)} "
            f"(salto de PSI de {salto_max:.3f} respecto al período anterior)."
        )
    else:
        st.success("✅ No se detectaron cambios abruptos entre períodos consecutivos.")


# ------------------------------------------------------------------
# TAB 3: Recomendaciones
# ------------------------------------------------------------------
with tab3:
    st.subheader("Recomendaciones automáticas")

    recomendaciones = generar_recomendaciones(reporte)
    for r in recomendaciones:
        if r.startswith("🔴"):
            st.error(r)
        elif r.startswith("🟡"):
            st.warning(r)
        else:
            st.success(r)

    st.divider()
    st.markdown(
        """
        **Criterio general de acción:**
        - 🔴 **Crítico (PSI > 0.25 o equivalente):** el modelo debería re-entrenarse
          con datos recientes antes de seguir usándose en producción — la población
          actual difiere lo suficiente de la que vio el modelo en entrenamiento
          como para que sus predicciones dejen de ser confiables.
        - 🟡 **Alerta (PSI entre 0.10 y 0.25):** no amerita re-entrenamiento
          inmediato, pero conviene monitorear de cerca en el próximo ciclo y
          revisar si las métricas de negocio (Recall, Precision) también se
          están degradando.
        - 🟢 **Estable:** sin acción requerida.
        """
    )
