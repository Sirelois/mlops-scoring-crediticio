"""
model_monitoring.py
Monitoreo y detección de Data Drift para el modelo de Scoring Crediticio (PI M5).

Compara la distribución de la población "histórica" (créditos más antiguos)
contra la población "actual" (créditos más recientes) para detectar cambios
que puedan afectar el desempeño del modelo entrenado en el Avance 2.

Métricas implementadas:
- Kolmogorov-Smirnov (KS test): variables numéricas
- Population Stability Index (PSI): variables numéricas
- Jensen-Shannon divergence: variables numéricas
- Chi-cuadrado: variables categóricas

La separación histórico/actual se hace por fecha_prestamo, simulando el
muestreo periódico que se haría en producción con datos que van llegando
en el tiempo (misma lógica temporal que en el bloque de MLflow de la
lectura del módulo).
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from pathlib import Path

from ft_engineering import cargar_datos, COLUMNAS_NUMERICAS, COLUMNAS_CATEGORICAS_NOMINALES, COLUMNAS_CATEGORICAS_ORDINALES

# ------------------------------------------------------------------
# Umbrales de alerta (convenciones estándar de la industria)
# ------------------------------------------------------------------
UMBRAL_PSI_ALERTA = 0.10      # PSI > 0.10: cambio moderado; > 0.25: cambio importante
UMBRAL_PSI_CRITICO = 0.25
UMBRAL_KS_PVALUE = 0.05       # p-value < 0.05: distribuciones significativamente distintas
UMBRAL_JS_ALERTA = 0.10
UMBRAL_CHI2_PVALUE = 0.05


def cargar_datos_para_monitoreo(path_csv: str) -> pd.DataFrame:
    """Carga y limpia los datos (reutilizando la lógica del Avance 2)."""
    df = cargar_datos(path_csv)
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
    return df.sort_values("fecha_prestamo").reset_index(drop=True)


def dividir_historico_actual(df: pd.DataFrame, percentil_corte: float = 0.7):
    """
    Divide el dataset en 'histórico' (población de referencia) y 'actual'
    (población reciente a monitorear), usando fecha_prestamo.

    percentil_corte=0.7 significa que el 70% de los créditos más antiguos
    forman la referencia, y el 30% más reciente es lo que se audita.
    """
    fecha_corte = df["fecha_prestamo"].quantile(percentil_corte)
    historico = df[df["fecha_prestamo"] < fecha_corte].copy()
    actual = df[df["fecha_prestamo"] >= fecha_corte].copy()
    return historico, actual, fecha_corte


def calcular_psi(referencia: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index. Divide la variable de referencia en `bins`
    grupos de igual frecuencia (percentiles), y compara qué proporción de
    cada grupo cae en referencia vs. actual.

    Interpretación: <0.10 sin cambio relevante, 0.10-0.25 cambio moderado,
    >0.25 cambio importante (revisar el modelo).
    """
    referencia = referencia.dropna()
    actual = actual.dropna()
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan

    breakpoints = np.linspace(0, 100, bins + 1)
    cortes = np.percentile(referencia, breakpoints)
    cortes[0], cortes[-1] = -np.inf, np.inf
    cortes = np.unique(cortes)  # evita cortes duplicados si hay muchos valores repetidos

    dist_ref = pd.cut(referencia, cortes).value_counts(normalize=True).sort_index()
    dist_act = pd.cut(actual, cortes).value_counts(normalize=True).sort_index()

    dist_ref = dist_ref.replace(0, 1e-6)
    dist_act = dist_act.reindex(dist_ref.index, fill_value=1e-6).replace(0, 1e-6)

    psi = ((dist_act - dist_ref) * np.log(dist_act / dist_ref)).sum()
    return float(psi)


def calcular_jensen_shannon(referencia: pd.Series, actual: pd.Series, bins: int = 20) -> float:
    """
    Jensen-Shannon divergence entre las distribuciones de referencia y actual,
    usando un histograma compartido. Valor entre 0 (idénticas) y 1 (totalmente
    distintas).
    """
    from scipy.spatial.distance import jensenshannon

    referencia = referencia.dropna()
    actual = actual.dropna()
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan

    minimo = min(referencia.min(), actual.min())
    maximo = max(referencia.max(), actual.max())
    if minimo == maximo:
        return 0.0

    limites = np.linspace(minimo, maximo, bins + 1)
    hist_ref, _ = np.histogram(referencia, bins=limites, density=True)
    hist_act, _ = np.histogram(actual, bins=limites, density=True)

    hist_ref = hist_ref + 1e-10
    hist_act = hist_act + 1e-10

    return float(jensenshannon(hist_ref, hist_act))


def evaluar_drift_numericas(historico: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Calcula KS, PSI y Jensen-Shannon para cada variable numérica."""
    filas = []
    for col in COLUMNAS_NUMERICAS:
        ref_col = historico[col]
        act_col = actual[col]

        ks_stat, ks_pvalue = ks_2samp(ref_col.dropna(), act_col.dropna())
        psi_val = calcular_psi(ref_col, act_col)
        js_val = calcular_jensen_shannon(ref_col, act_col)

        alerta = (
            (ks_pvalue < UMBRAL_KS_PVALUE)
            or (psi_val > UMBRAL_PSI_ALERTA)
            or (js_val > UMBRAL_JS_ALERTA)
        )
        critico = (psi_val > UMBRAL_PSI_CRITICO)

        filas.append({
            "variable": col,
            "tipo": "numerica",
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "psi": round(psi_val, 4),
            "jensen_shannon": round(js_val, 4),
            "chi2_pvalue": np.nan,
            "alerta": alerta,
            "critico": critico,
        })
    return pd.DataFrame(filas)


def evaluar_drift_categoricas(historico: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Calcula Chi-cuadrado para cada variable categórica."""
    columnas_cat = COLUMNAS_CATEGORICAS_NOMINALES + COLUMNAS_CATEGORICAS_ORDINALES
    filas = []
    for col in columnas_cat:
        conteo_ref = historico[col].value_counts()
        conteo_act = actual[col].value_counts()
        categorias = sorted(set(conteo_ref.index) | set(conteo_act.index))

        tabla = pd.DataFrame({
            "historico": [conteo_ref.get(c, 0) for c in categorias],
            "actual": [conteo_act.get(c, 0) for c in categorias],
        }, index=categorias)

        # Chi2 necesita al menos 2 categorías con datos en ambos grupos
        if tabla.shape[0] < 2 or (tabla.sum(axis=1) == 0).any():
            chi2_pvalue = np.nan
        else:
            _, chi2_pvalue, _, _ = chi2_contingency(tabla)

        alerta = (chi2_pvalue < UMBRAL_CHI2_PVALUE) if not np.isnan(chi2_pvalue) else False

        filas.append({
            "variable": col,
            "tipo": "categorica",
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "psi": np.nan,
            "jensen_shannon": np.nan,
            "chi2_pvalue": round(chi2_pvalue, 4) if not np.isnan(chi2_pvalue) else np.nan,
            "alerta": alerta,
            "critico": False,
        })
    return pd.DataFrame(filas)


def generar_reporte_drift(historico: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Combina el análisis de numéricas y categóricas en una sola tabla resumen."""
    tabla_num = evaluar_drift_numericas(historico, actual)
    tabla_cat = evaluar_drift_categoricas(historico, actual)
    return pd.concat([tabla_num, tabla_cat], ignore_index=True)


def drift_evolucion_temporal(df: pd.DataFrame, variable: str, n_periodos: int = 6) -> pd.DataFrame:
    """
    Calcula cómo evoluciona el PSI de una variable a lo largo del tiempo,
    dividiendo el dataset en n_periodos ventanas cronológicas y comparando
    cada ventana contra la primera (que actúa como referencia fija).

    Sirve para la sección "Análisis temporal" de la app de Streamlit.
    """
    df = df.dropna(subset=["fecha_prestamo"]).sort_values("fecha_prestamo")
    df["periodo"] = pd.qcut(df["fecha_prestamo"].rank(method="first"), n_periodos, labels=False)

    referencia = df[df["periodo"] == 0][variable]
    filas = []
    for periodo in sorted(df["periodo"].unique()):
        ventana = df[df["periodo"] == periodo]
        psi_val = calcular_psi(referencia, ventana[variable])
        fecha_media = ventana["fecha_prestamo"].mean()
        filas.append({"periodo": periodo, "fecha": fecha_media, "psi": psi_val})

    return pd.DataFrame(filas)


def generar_recomendaciones(tabla_drift: pd.DataFrame) -> list:
    """
    Genera mensajes automáticos de alerta/recomendación en base a la tabla
    de resultados de drift, tal como pide la consigna.
    """
    recomendaciones = []

    variables_criticas = tabla_drift[tabla_drift["critico"] == True]["variable"].tolist()
    variables_alerta = tabla_drift[
        (tabla_drift["alerta"] == True) & (tabla_drift["critico"] == False)
    ]["variable"].tolist()

    if variables_criticas:
        recomendaciones.append(
            f"🔴 CRÍTICO: las variables {variables_criticas} muestran un cambio "
            f"importante en su distribución (PSI > {UMBRAL_PSI_CRITICO}). "
            f"Se recomienda reentrenar el modelo con datos recientes antes de "
            f"seguir usándolo en producción."
        )
    if variables_alerta:
        recomendaciones.append(
            f"🟡 ALERTA: las variables {variables_alerta} muestran cambios "
            f"moderados. Se recomienda monitorear de cerca en el próximo ciclo "
            f"y revisar si el desempeño del modelo se está degradando."
        )
    if not variables_criticas and not variables_alerta:
        recomendaciones.append(
            "🟢 Sin alertas: no se detectaron cambios significativos en la "
            "población respecto a la referencia histórica."
        )
    return recomendaciones


if __name__ == "__main__":
    ruta_csv = Path(__file__).resolve().parent.parent.parent / "Base_de_datos.csv"
    df = cargar_datos_para_monitoreo(str(ruta_csv))
    historico, actual, fecha_corte = dividir_historico_actual(df)

    print(f"Fecha de corte: {fecha_corte}")
    print(f"Histórico: {len(historico)} registros | Actual: {len(actual)} registros\n")

    reporte = generar_reporte_drift(historico, actual)
    print("=== Reporte de Data Drift ===")
    print(reporte.to_string(index=False))

    print("\n=== Recomendaciones ===")
    for r in generar_recomendaciones(reporte):
        print(r)
