"""
model_training_evaluation.py
Entrenamiento y evaluación de modelos supervisados para el Scoring Crediticio (PI M5).

Entrena varios modelos de clasificación (con SMOTE para compensar el desbalance
95/5 detectado en el EDA), los compara con métricas apropiadas para clases
desbalanceadas (Recall, F1, AUC — no Accuracy), y selecciona el de mejor
performance.

Reutiliza dos funciones para los procesos repetidos, tal como pide la consigna:
- build_model(): entrena un modelo dado y devuelve el pipeline ya entrenado
- summarize_classification(): calcula y devuelve las métricas de evaluación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from ft_engineering import generar_datasets, construir_pipeline_preprocesamiento


# ------------------------------------------------------------------
# Modelos a comparar. Todos con class_weight/parametros por default;
# el balanceo real lo hace SMOTE dentro del pipeline, no los modelos.
# ------------------------------------------------------------------
MODELOS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, solver="liblinear"),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "xgboost": XGBClassifier(random_state=42, eval_metric="logloss"),
}


def build_model(nombre_modelo: str, modelo, preprocesador, X_train, y_train):
    """
    Arma un pipeline completo (preprocesamiento + SMOTE + modelo) y lo entrena.
    Se usa imblearn.Pipeline (no sklearn.Pipeline) porque necesita poder
    incluir un paso de resampling (SMOTE), algo que el Pipeline de sklearn
    no soporta de forma nativa.
    """
    pipeline = ImbPipeline([
        ("preprocesamiento", preprocesador),
        ("smote", SMOTE(random_state=42)),
        ("modelo", modelo),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def summarize_classification(nombre_modelo: str, pipeline, X_test, y_test) -> dict:
    """
    Evalúa un pipeline ya entrenado sobre el conjunto de test y devuelve
    un diccionario con las métricas relevantes.

    IMPORTANTE: precision/recall/f1 se calculan sobre la clase 0 (NO paga a
    tiempo), no sobre la clase 1 (default de sklearn). La clase 0 es la
    minoritaria (5% del dataset) y es la que representa el riesgo real que
    el negocio necesita anticipar — un modelo con alto Recall en la clase
    mayoritaria (paga a tiempo) puede seguir siendo inútil si no detecta
    a los clientes que no van a pagar.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metricas = {
        "modelo": nombre_modelo,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_no_paga": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall_no_paga": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1_no_paga": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }
    return metricas


def comparar_modelos(X_train, X_test, y_train, y_test, preprocesador) -> pd.DataFrame:
    """
    Entrena y evalúa todos los modelos definidos en MODELOS, devolviendo
    una tabla resumen ordenada por AUC (métrica robusta al desbalance).
    """
    resultados = []
    pipelines_entrenados = {}

    for nombre, modelo in MODELOS.items():
        print(f"Entrenando {nombre}...")
        pipeline = build_model(nombre, modelo, preprocesador, X_train, y_train)
        metricas = summarize_classification(nombre, pipeline, X_test, y_test)
        resultados.append(metricas)
        pipelines_entrenados[nombre] = pipeline

    tabla = pd.DataFrame(resultados).sort_values("recall_no_paga", ascending=False).reset_index(drop=True)
    return tabla, pipelines_entrenados


def graficar_comparacion(tabla: pd.DataFrame):
    """Gráfico comparativo de barras para las métricas clave de cada modelo."""
    metricas_a_graficar = ["recall_no_paga", "f1_no_paga", "auc"]
    tabla_plot = tabla.set_index("modelo")[metricas_a_graficar]

    tabla_plot.plot(kind="bar", figsize=(10, 5))
    plt.title("Comparación de modelos: Recall, F1 y AUC")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.show()


def graficar_matriz_confusion(pipeline, X_test, y_test, nombre_modelo: str):
    """Matriz de confusión del modelo seleccionado como mejor."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No paga (0)", "Paga (1)"],
                yticklabels=["No paga (0)", "Paga (1)"])
    plt.title(f"Matriz de confusión — {nombre_modelo}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    ruta_csv = Path(__file__).resolve().parent.parent.parent / "Base_de_datos.csv"
    X_train, X_test, y_train, y_test, preprocesador = generar_datasets(str(ruta_csv))

    tabla_resultados, pipelines = comparar_modelos(
        X_train, X_test, y_train, y_test, preprocesador
    )

    print("\n=== Tabla resumen de evaluación ===")
    print(tabla_resultados.round(3))

    mejor_modelo_nombre = tabla_resultados.iloc[0]["modelo"]
    print(f"\nMejor modelo según Recall de la clase 'no paga': {mejor_modelo_nombre}")

    graficar_comparacion(tabla_resultados)
    graficar_matriz_confusion(pipelines[mejor_modelo_nombre], X_test, y_test, mejor_modelo_nombre)

    print("\n=== Classification report del mejor modelo ===")
    y_pred_mejor = pipelines[mejor_modelo_nombre].predict(X_test)
    print(classification_report(y_test, y_pred_mejor, target_names=["No paga (0)", "Paga (1)"]))

    # ------------------------------------------------------------------
    # Justificación de la selección del modelo
    # ------------------------------------------------------------------
    # Se selecciona Logistic Regression como mejor modelo, priorizando Recall
    # de la clase "No paga" (0) por sobre AUC o Accuracy. Los modelos basados
    # en árboles/boosting (Decision Tree, Random Forest, Gradient Boosting,
    # XGBoost) obtienen mejor AUC y Accuracy, pero detectan menos del 8% de
    # los clientes que efectivamente no pagan (Recall < 0.08 en todos ellos),
    # volviéndolos inútiles como herramienta de anticipación de riesgo pese
    # a sus métricas globales altas.
    #
    # Logistic Regression detecta el 45% de los no-pagos reales (vs 2-8% del
    # resto), a costa de una Precision baja (7%): genera bastantes falsos
    # positivos (clientes marcados como riesgo que en realidad pagarían).
    # Este trade-off se considera aceptable para un modelo de scoring
    # crediticio, donde el costo de no detectar un impago (falso negativo)
    # suele ser mayor al costo de una alerta de más (falso positivo) que
    # luego puede revisarse con otros criterios antes de rechazar el crédito.
    #
    # Próximo paso natural (fuera del alcance de este avance): ajustar el
    # umbral de decisión (actualmente 0.5 por default) para balancear mejor
    # Recall y Precision según el apetito de riesgo del negocio.
    print(f"\n{'='*70}")
    print(f"MODELO SELECCIONADO: {mejor_modelo_nombre}")
    print(f"{'='*70}")
    print("Justificación: prioriza Recall de la clase minoritaria (no paga)")
    print("por sobre AUC/Accuracy, ya que es la métrica alineada con el")
    print("objetivo de negocio (anticipar riesgo de impago). Ver comentario")
    print("en el código para el detalle completo del trade-off.")
