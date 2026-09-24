"""
model_training_evaluation.py  (v2)
Entrenamiento, selección y evaluación de modelos para el Scoring Crediticio (PI M5).

Cambios respecto de la v1 (y por qué):
1. Selección por PR-AUC de la clase "no paga" con validación cruzada (5 folds),
   no por Recall con umbral 0.5. El recall depende del umbral, así que comparar
   recalls a 0.5 entre modelos distintos no mide qué modelo discrimina mejor.
2. Desbalance tratado con pesos de clase (class_weight / scale_pos_weight) en
   lugar de SMOTE: SMOTE interpola sobre variables one-hot y genera clientes
   sintéticos poco realistas.
3. Logistic Regression ahora con StandardScaler (en la v1 las variables
   numéricas entraban sin escalar).
4. Optimización de hiperparámetros (RandomizedSearchCV) de los 2 mejores modelos.
5. Umbral de decisión elegido con un criterio de negocio (recall objetivo)
   sobre predicciones out-of-fold de train. El test se usa una sola vez.
6. El modelo final se guarda con joblib para el deploy (Avance 4).

Convención: Pago_atiempo = 1 (paga), 0 (no paga). La clase de riesgo es 0.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from ft_engineering import generar_datasets


# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]          # src -> mlops_pipeline -> repo
RUTA_CSV = RAIZ / "Base_de_datos.csv"
RUTA_SALIDA = Path(__file__).resolve().parent
ARCHIVO_MODELO = RUTA_SALIDA / "modelo_scoring.joblib"
ARCHIVO_GRAFICO = RUTA_SALIDA / "comparacion_modelos_cv.png"

CLASE_RIESGO = 0            # "no paga"
RECALL_OBJETIVO = 0.60      # % de morosos que el negocio quiere detectar
N_ITER_BUSQUEDA = 25
SEMILLA = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


# ------------------------------------------------------------------
# Métricas centradas en la clase "no paga"
# ------------------------------------------------------------------
def prob_no_paga(estimator, X) -> np.ndarray:
    """Probabilidad de NO pagar (score de riesgo)."""
    idx = list(estimator.classes_).index(CLASE_RIESGO)
    return estimator.predict_proba(X)[:, idx]


def _es_riesgo(y) -> np.ndarray:
    return (np.asarray(y) == CLASE_RIESGO).astype(int)


def pr_auc_no_paga(estimator, X, y) -> float:
    return average_precision_score(_es_riesgo(y), prob_no_paga(estimator, X))


def roc_auc_no_paga(estimator, X, y) -> float:
    return roc_auc_score(_es_riesgo(y), prob_no_paga(estimator, X))


SCORING = {"pr_auc": pr_auc_no_paga, "roc_auc": roc_auc_no_paga}


# ------------------------------------------------------------------
# Modelos y espacios de búsqueda
# ------------------------------------------------------------------
def definir_modelos(y_train) -> dict:
    n_riesgo = (np.asarray(y_train) == CLASE_RIESGO).sum()
    n_resto = len(y_train) - n_riesgo
    # scale_pos_weight pondera la clase 1 (paga). Como la minoritaria es la 0,
    # se usa un valor < 1: baja el peso de la mayoritaria (equivale a balancear).
    spw = n_riesgo / n_resto
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEMILLA),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=SEMILLA),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEMILLA),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEMILLA),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=spw, eval_metric="logloss",
            n_jobs=-1, random_state=SEMILLA),
    }


ESPACIOS = {
    "logistic_regression": {
        "modelo__C": loguniform(1e-3, 1e2),
    },
    "decision_tree": {
        "modelo__max_depth": randint(2, 12),
        "modelo__min_samples_leaf": randint(5, 100),
    },
    "random_forest": {
        "modelo__max_depth": [None, 5, 8, 12, 16],
        "modelo__min_samples_leaf": randint(1, 30),
        "modelo__max_features": ["sqrt", 0.3, 0.5],
    },
    "hist_gradient_boosting": {
        "modelo__learning_rate": loguniform(0.01, 0.3),
        "modelo__max_depth": [None, 3, 5, 8],
        "modelo__max_leaf_nodes": randint(8, 64),
        "modelo__min_samples_leaf": randint(10, 100),
        "modelo__l2_regularization": loguniform(1e-3, 10),
    },
    "xgboost": {
        "modelo__n_estimators": randint(100, 600),
        "modelo__learning_rate": loguniform(0.01, 0.3),
        "modelo__max_depth": randint(2, 8),
        "modelo__subsample": uniform(0.6, 0.4),
        "modelo__colsample_bytree": uniform(0.5, 0.5),
        "modelo__min_child_weight": randint(1, 20),
    },
}


# ------------------------------------------------------------------
# Funciones reutilizables (consigna: build_model / summarize_classification)
# ------------------------------------------------------------------
def build_model(nombre_modelo: str, modelo, preprocesador) -> Pipeline:
    """Arma el pipeline preprocesamiento (+ escalado si es lineal) + modelo, sin entrenar."""
    # sparse_threshold=0 fuerza salida densa (HistGradientBoosting no acepta sparse)
    pasos = [("preprocesamiento", clone(preprocesador).set_params(sparse_threshold=0))]
    if nombre_modelo == "logistic_regression":
        pasos.append(("escalado", StandardScaler()))
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


def summarize_classification(nombre_modelo: str, pipeline, X_test, y_test, umbral: float) -> dict:
    """Métricas de negocio sobre test para un umbral dado de probabilidad de no pago."""
    score = prob_no_paga(pipeline, X_test)
    y_riesgo = _es_riesgo(y_test)
    marcado = (score >= umbral).astype(int)

    tasa_base = y_riesgo.mean()
    precision = precision_score(y_riesgo, marcado, zero_division=0)
    return {
        "modelo": nombre_modelo,
        "umbral": round(float(umbral), 4),
        "roc_auc": roc_auc_score(y_riesgo, score),
        "pr_auc": average_precision_score(y_riesgo, score),
        "recall_no_paga": recall_score(y_riesgo, marcado, zero_division=0),
        "precision_no_paga": precision,
        "lift": precision / tasa_base if tasa_base > 0 else np.nan,
        "clientes_marcados": int(marcado.sum()),
        "pct_marcados": marcado.mean(),
        "morosos_detectados": int((marcado & y_riesgo).sum()),
        "morosos_totales": int(y_riesgo.sum()),
        "detectados_si_fuera_azar": round(marcado.mean() * y_riesgo.sum(), 1),
        "buenos_clientes_rechazados": int((marcado & (1 - y_riesgo)).sum()),
    }


# ------------------------------------------------------------------
# Etapas
# ------------------------------------------------------------------
def comparar_modelos_cv(modelos: dict, preprocesador, X_train, y_train) -> pd.DataFrame:
    filas = []
    for nombre, modelo in modelos.items():
        print(f"  CV {nombre}...")
        res = cross_validate(build_model(nombre, modelo, preprocesador),
                             X_train, y_train, cv=CV, scoring=SCORING, n_jobs=1)
        filas.append({
            "modelo": nombre,
            "pr_auc_cv": res["test_pr_auc"].mean(),
            "pr_auc_std": res["test_pr_auc"].std(),
            "roc_auc_cv": res["test_roc_auc"].mean(),
        })
    return pd.DataFrame(filas).sort_values("pr_auc_cv", ascending=False).reset_index(drop=True)


def graficar_comparacion(tabla: pd.DataFrame):
    tabla.set_index("modelo")[["pr_auc_cv", "roc_auc_cv"]].plot(kind="bar", figsize=(10, 5))
    plt.title("Comparación de modelos (validación cruzada, clase 'no paga')")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(ARCHIVO_GRAFICO, dpi=120)
    plt.close()


def optimizar(nombre: str, modelo, preprocesador, X_train, y_train) -> RandomizedSearchCV:
    busqueda = RandomizedSearchCV(
        build_model(nombre, modelo, preprocesador), ESPACIOS[nombre],
        n_iter=N_ITER_BUSQUEDA, scoring=SCORING, refit="pr_auc",
        cv=CV, random_state=SEMILLA, n_jobs=1, verbose=0,
    )
    busqueda.fit(X_train, y_train)
    return busqueda


def elegir_umbral(pipeline, X_train, y_train, recall_objetivo: float):
    """Umbral más alto que alcanza el recall objetivo, sobre predicciones out-of-fold."""
    idx = list(np.unique(y_train)).index(CLASE_RIESGO)
    proba_oof = cross_val_predict(clone(pipeline), X_train, y_train,
                                  cv=CV, method="predict_proba")[:, idx]
    precision, recall, umbrales = precision_recall_curve(_es_riesgo(y_train), proba_oof)
    i = np.where(recall[:-1] >= recall_objetivo)[0][-1]
    return umbrales[i], precision[i], recall[i]


def marcados_para_recall(score, y_riesgo, recall_ref: float) -> int:
    """Cuántos clientes hay que marcar para detectar el recall_ref de los morosos."""
    scores_morosos = np.sort(score[y_riesgo == 1])[::-1]
    k = int(np.ceil(recall_ref * len(scores_morosos)))
    return int((score >= scores_morosos[k - 1]).sum())


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test, preprocesador = generar_datasets(str(RUTA_CSV))
    tasa = (np.asarray(y_train) == CLASE_RIESGO).mean()
    print(f"Train {X_train.shape} | Test {X_test.shape}")
    print(f"Tasa de 'no paga': {tasa:.3f}  -> PR-AUC de un modelo al azar ~ {tasa:.3f}")

    modelos = definir_modelos(y_train)

    print("\n[1/4] Comparación por validación cruzada (5 folds, solo train)")
    tabla_cv = comparar_modelos_cv(modelos, preprocesador, X_train, y_train)
    print(tabla_cv.round(3).to_string(index=False))
    graficar_comparacion(tabla_cv)

    print("\n[2/4] Optimización de hiperparámetros (2 mejores)")
    busquedas = {}
    for nombre in tabla_cv["modelo"].head(2):
        print(f"  Buscando {nombre} ({N_ITER_BUSQUEDA} combinaciones x 5 folds)...")
        busquedas[nombre] = optimizar(nombre, modelos[nombre], preprocesador, X_train, y_train)
        print(f"    PR-AUC CV optimizado: {busquedas[nombre].best_score_:.3f}")
    ganador = max(busquedas, key=lambda n: busquedas[n].best_score_)
    mejor = busquedas[ganador].best_estimator_
    print(f"  Ganador: {ganador}")
    print(f"  Hiperparámetros: {busquedas[ganador].best_params_}")

    print(f"\n[3/4] Umbral de decisión (recall objetivo {RECALL_OBJETIVO:.0%}, out-of-fold en train)")
    umbral, p_oof, r_oof = elegir_umbral(mejor, X_train, y_train, RECALL_OBJETIVO)
    print(f"  Umbral: {umbral:.4f} | recall OOF {r_oof:.3f} | precision OOF {p_oof:.3f}")

    print("\n[4/4] Evaluación final en test (una sola vez)")
    metricas = summarize_classification(ganador, mejor, X_test, y_test, umbral)
    for k, v in metricas.items():
        print(f"  {k:28s} {v:.3f}" if isinstance(v, float) else f"  {k:28s} {v}")

    score = prob_no_paga(mejor, X_test)
    ref = marcados_para_recall(score, _es_riesgo(y_test), 0.48)
    print(f"\n  Referencia vs v1: para detectar ~48% de morosos, este modelo marca "
          f"{ref} clientes (Logistic Regression v1 marcaba ~910).")

    y_pred = np.where(score >= umbral, CLASE_RIESGO, 1)
    print("\n  Matriz de confusión (filas: real 0/1, columnas: predicho 0/1)")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print(classification_report(y_test, y_pred, labels=[0, 1],
                                target_names=["No paga (0)", "Paga (1)"], zero_division=0))

    joblib.dump({
        "pipeline": mejor,
        "umbral": float(umbral),
        "clase_riesgo": CLASE_RIESGO,
        "modelo": ganador,
        "columnas": list(X_train.columns),
        "metricas_test": metricas,
    }, ARCHIVO_MODELO)
    print(f"Modelo guardado en: {ARCHIVO_MODELO}")
    print(f"Gráfico guardado en: {ARCHIVO_GRAFICO}")

    imprimir_justificacion(ganador, busquedas[ganador].best_score_, metricas, ref)


def imprimir_justificacion(ganador: str, pr_auc_cv: float, m: dict, marcados_ref: int):
    """
    ------------------------------------------------------------------
    Justificación de la selección
    ------------------------------------------------------------------
    Por qué NO se elige por Recall con umbral 0.5 (criterio de la v1):
    el recall depende del umbral. Comparar recalls a 0.5 entre modelos solo
    muestra cuál tiene las probabilidades más corridas hacia "no paga", no cuál
    distingue mejor morosos de buenos pagadores. En la v1 esto llevó a elegir
    Logistic Regression, que para detectar 48% de morosos marcaba ~910 de 2153
    clientes (lift ~1.15, casi azar).

    Criterio de la v2:
    1) Se elige el modelo que mejor ORDENA a los clientes por riesgo
        (PR-AUC de la clase "no paga", validación cruzada en train). PR-AUC es
        preferible a ROC-AUC con 95/5 de desbalance porque se concentra en la
        clase minoritaria.
    2) El umbral se fija DESPUÉS, con un criterio de negocio (recall objetivo),
        usando predicciones out-of-fold para no contaminar el test.

    Limitación: la señal de los datos es débil (ROC-AUC ~0.67). El modelo sirve
    para priorizar solicitudes a revisión manual, no para rechazar automáticamente.
    """
    tasa_base = m["morosos_totales"] / (m["clientes_marcados"] / m["pct_marcados"])
    print("\n" + "=" * 70)
    print(f"MODELO SELECCIONADO: {ganador}")
    print("=" * 70)
    print("Criterio: mejor PR-AUC de la clase 'no paga' en validación cruzada")
    print(f"(PR-AUC CV {pr_auc_cv:.3f} | test {m['pr_auc']:.3f} | azar ~{tasa_base:.3f}).")
    print(f"Umbral de negocio {m['umbral']}: detecta {m['morosos_detectados']} de "
          f"{m['morosos_totales']} morosos ({m['recall_no_paga']:.0%}), marcando "
          f"{m['pct_marcados']:.0%} de las solicitudes (lift {m['lift']:.2f}).")
    print(f"Frente a la v1: para el mismo 48% de morosos marca {marcados_ref} "
          f"clientes en lugar de ~910.")
    print("Uso recomendado: priorizar solicitudes para revisión manual,")
    print("no rechazo automático (señal débil en los datos, ROC-AUC ~0.67).")


if __name__ == "__main__":
    main()
