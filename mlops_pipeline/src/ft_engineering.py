"""
ft_engineering.py
Ingeniería de características para el modelo de Scoring Crediticio (PI M5).

Genera el pipeline de transformación de datos (imputación + encoding) y
devuelve los conjuntos de entrenamiento y evaluación listos para modelar.

Basado en los hallazgos del EDA (comprension_eda.ipynb):
- 'puntaje' se excluye: data leakage confirmado (correlación 0.88 con el target,
  separa perfectamente las clases, no estaría disponible en producción).
- 'fecha_prestamo' se excluye como predictor directo.
- Se descarta una variable de cada par con colinealidad alta (>0.7):
  cuota_pactada (vs capital_prestado), creditos_sectorFinanciero (vs cant_creditosvigentes),
  saldo_principal (vs saldo_total).
- 'tipo_credito' se trata como categórica (son códigos, no cantidades).
- 'tendencia_ingresos' se trata como categórica ORDINAL (Decreciente < Estable < Creciente).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ------------------------------------------------------------------
# Columnas excluidas del modelo (ver docstring / EDA)
# ------------------------------------------------------------------
COLUMNAS_EXCLUIR = [
    "puntaje",              # data leakage
    "fecha_prestamo",       # no se usa como predictor directo
    "cuota_pactada",        # colineal con capital_prestado
    "creditos_sectorFinanciero",  # colineal con cant_creditosvigentes
    "saldo_principal",      # colineal con saldo_total
]

TARGET = "Pago_atiempo"

# Orden lógico para la variable ordinal
ORDEN_TENDENCIA = [["Decreciente", "Estable", "Creciente"]]

COLUMNAS_NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "puntaje_datacredito", "cant_creditosvigentes",
    "huella_consulta", "saldo_mora", "saldo_total", "saldo_mora_codeudor",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]

COLUMNAS_CATEGORICAS_NOMINALES = ["tipo_laboral", "tipo_credito"]
COLUMNAS_CATEGORICAS_ORDINALES = ["tendencia_ingresos"]


def cargar_datos(path_csv: str) -> pd.DataFrame:
    """Carga el CSV y aplica las mismas correcciones detectadas en el EDA."""
    df = pd.read_csv(path_csv)

    # Unificar marcadores de nulos (igual que en comprension_eda.ipynb)
    marcadores_nulos = ["", " ", "NaN", "nan", "N/A", "n/a", "-", "None"]
    df = df.replace(marcadores_nulos, np.nan)

    # tendencia_ingresos: solo 3 categorías válidas
    categorias_validas = ["Estable", "Creciente", "Decreciente"]
    df.loc[~df["tendencia_ingresos"].isin(categorias_validas), "tendencia_ingresos"] = np.nan

    # edad_cliente / salario_cliente: outliers de carga (edad > 90)
    df.loc[df["edad_cliente"] > 90, ["edad_cliente", "salario_cliente"]] = np.nan

    # salario_cliente: outliers adicionales
    df.loc[df["salario_cliente"] > 100_000_000, "salario_cliente"] = np.nan

    # puntaje: se excluye más abajo (leakage), no hace falta corregirla acá

    # tipo_credito como texto, para que se trate como categórica
    df["tipo_credito"] = df["tipo_credito"].astype(str)

    return df


def separar_x_y(df: pd.DataFrame):
    """Separa en X (predictoras) e y (target), excluyendo columnas no deseadas."""
    y = df[TARGET]
    columnas_a_excluir = COLUMNAS_EXCLUIR + [TARGET]
    X = df.drop(columns=[c for c in columnas_a_excluir if c in df.columns])
    return X, y


def construir_pipeline_preprocesamiento() -> ColumnTransformer:
    """
    Arma el ColumnTransformer según el esquema pedido en la consigna:
    numeric -> SimpleImputer
    categoric -> SimpleImputer + OneHotEncoder
    categoric ordinales -> SimpleImputer + OrdinalEncoder
    """
    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    pipeline_categorico_nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    pipeline_categorico_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ORDEN_TENDENCIA)),
    ])

    preprocesador = ColumnTransformer([
        ("num", pipeline_numerico, COLUMNAS_NUMERICAS),
        ("cat_nom", pipeline_categorico_nominal, COLUMNAS_CATEGORICAS_NOMINALES),
        ("cat_ord", pipeline_categorico_ordinal, COLUMNAS_CATEGORICAS_ORDINALES),
    ])

    return preprocesador


def generar_datasets(path_csv: str, test_size: float = 0.2, random_state: int = 42):
    """
    Función principal: carga, limpia, separa X/y, hace el split y devuelve
    todo listo para entrenar. El ColumnTransformer se devuelve sin fitear
    (se fitea recién dentro del pipeline de modelado, sobre X_train).
    """
    df = cargar_datos(path_csv)
    X, y = separar_x_y(df)

    # Split estratificado: fundamental dado el desbalance 95/5 detectado en el EDA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocesador = construir_pipeline_preprocesamiento()

    return X_train, X_test, y_train, y_test, preprocesador


if __name__ == "__main__":
    # Ruta calculada desde la ubicación del propio script (__file__), no desde
    # el directorio actual de la terminal (CWD) — así funciona sin importar
    # desde dónde se ejecute (terminal en pi/, en src/, botón "Run" de VSCode, etc.)
    ruta_csv = Path(__file__).resolve().parent.parent.parent / "Base_de_datos.csv"

    X_train, X_test, y_train, y_test, preprocesador = generar_datasets(str(ruta_csv))
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Balance train: \n{y_train.value_counts(normalize=True).round(3)}")
