"""
model_deploy.py
API REST (FastAPI) para disponibilizar el modelo de Scoring Crediticio.

Carga el modelo entrenado en model_training_evaluation.py (modelo_scoring.joblib),
que incluye el pipeline completo (preprocesamiento + Random Forest) y el umbral
de negocio. Recibe uno o varios clientes y devuelve la probabilidad de no pago y
una recomendación.

Endpoints:
- GET  /          Estado de la API y datos del modelo cargado
- GET  /columnas  Variables que espera el modelo
- POST /predict   Predicción por lotes

Ejecución local (desde la raíz del repo):
    uvicorn model_deploy:app --app-dir mlops_pipeline/src --reload
Documentación interactiva: http://localhost:8000/docs
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


RUTA_MODELO = Path(__file__).resolve().parent / "modelo_scoring.joblib"

try:
    ARTEFACTO = joblib.load(RUTA_MODELO)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"No se encontró el modelo en {RUTA_MODELO}. "
        "Ejecutá primero model_training_evaluation.py."
    ) from exc

PIPELINE = ARTEFACTO["pipeline"]
UMBRAL = ARTEFACTO["umbral"]
CLASE_RIESGO = ARTEFACTO["clase_riesgo"]
COLUMNAS = ARTEFACTO["columnas"]
IDX_RIESGO = list(PIPELINE.classes_).index(CLASE_RIESGO)


def _tipos_de_columnas():
    """Lee del preprocesador qué columnas son numéricas y qué categorías acepta cada ordinal."""
    numericas, ordinales = [], {}
    preprocesador = PIPELINE.named_steps["preprocesamiento"]
    for nombre, transformador, columnas in preprocesador.transformers_:
        if nombre == "num":
            numericas.extend(columnas)
        elif nombre == "cat_ord":
            encoder = transformador.named_steps["encoder"]
            ordinales.update(dict(zip(columnas, encoder.categories_)))
    return numericas, ordinales


COLUMNAS_NUMERICAS, CATEGORIAS_ORDINALES = _tipos_de_columnas()


app = FastAPI(
    title="API de Scoring Crediticio",
    description=(
        "Estima la probabilidad de que un cliente no pague a tiempo. "
        "Uso recomendado: priorizar solicitudes para revisión manual, "
        "no rechazo automático."
    ),
    version="1.3.0",
)


class Solicitud(BaseModel):
    registros: list[dict[str, Any]] = Field(
        ..., min_length=1,
        description="Lista de clientes. Cada cliente es un objeto {variable: valor}. "
                    "Las variables esperadas se consultan en GET /columnas.",
    )


class Prediccion(BaseModel):
    indice: int
    probabilidad_no_pago: float
    riesgo_alto: bool
    recomendacion: str


class Respuesta(BaseModel):
    modelo: str
    umbral: float
    predicciones: list[Prediccion]
    advertencias: list[str]


def preparar_datos(registros: list[dict[str, Any]]):
    """Arma el DataFrame con las columnas del modelo y normaliza tipos."""
    df = pd.DataFrame(registros)
    advertencias = []

    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        advertencias.append(f"Variables ausentes, se imputan: {faltantes}")
        for c in faltantes:
            df[c] = np.nan

    sobrantes = [c for c in df.columns if c not in COLUMNAS]
    if sobrantes:
        advertencias.append(f"Variables ignoradas (el modelo no las usa): {sobrantes}")

    df = df[COLUMNAS].copy()

    for c in COLUMNAS_NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Categorías ordinales desconocidas -> nulo (las imputa el pipeline) en vez de error 500
    for c, categorias in CATEGORIAS_ORDINALES.items():
        invalidos = df[c].notna() & ~df[c].isin(categorias)
        if invalidos.any():
            advertencias.append(f"Valores no reconocidos en '{c}', se imputan.")
            df.loc[invalidos, c] = np.nan

    return df, advertencias


@app.get("/")
def estado():
    return {
        "estado": "ok",
        "modelo": ARTEFACTO["modelo"],
        "umbral": UMBRAL,
        "variables_esperadas": len(COLUMNAS),
        "metricas_test": {
            k: ARTEFACTO["metricas_test"][k]
            for k in ("roc_auc", "pr_auc", "recall_no_paga", "precision_no_paga", "lift")
        },
    }


@app.get("/columnas")
def columnas():
    return {
        "columnas": COLUMNAS,
        "numericas": COLUMNAS_NUMERICAS,
        "ordinales": {c: list(v) for c, v in CATEGORIAS_ORDINALES.items()},
    }


@app.post("/predict", response_model=Respuesta)
def predecir(solicitud: Solicitud):
    df, advertencias = preparar_datos(solicitud.registros)
    try:
        probabilidades = PIPELINE.predict_proba(df)[:, IDX_RIESGO]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo predecir: {exc}") from exc

    predicciones = [
        Prediccion(
            indice=i,
            probabilidad_no_pago=round(float(p), 4),
            riesgo_alto=bool(p >= UMBRAL),
            recomendacion="Derivar a revisión manual" if p >= UMBRAL else "Aprobación sugerida",
        )
        for i, p in enumerate(probabilidades)
    ]
    return Respuesta(
        modelo=ARTEFACTO["modelo"],
        umbral=UMBRAL,
        predicciones=predicciones,
        advertencias=advertencias,
    )
