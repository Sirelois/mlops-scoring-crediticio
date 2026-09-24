# MLOps – Modelo de Riesgo Crediticio

Proyecto Integrador del Módulo 5 (MLOps) – Data Science, Henry.

## 1. Caso de negocio

Una empresa financiera necesita anticipar el comportamiento de pago de nuevos usuarios a partir de información histórica de créditos. El objetivo es identificar, **antes de otorgar el crédito**, a los clientes con mayor probabilidad de **no pagar a tiempo**.

Los dos errores posibles no cuestan lo mismo:

- **Aprobar a un cliente que no paga** (falso negativo) genera pérdida directa de capital.
- **Rechazar a un cliente que sí paga** (falso positivo) implica perder un cliente e ingresos por intereses.

Como solo ~5% de los clientes no paga, un modelo que aprueba a todos tendría 95% de accuracy y no serviría para nada. Por eso el proyecto evalúa los modelos con métricas centradas en la clase minoritaria (**PR-AUC, Recall y Precision de "no paga"**) y no con accuracy.

## 2. Estructura del repositorio

```
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb               # Carga inicial de datos
│       ├── comprension_eda.ipynb            # Análisis exploratorio (EDA)
│       ├── ft_engineering.py                # Limpieza, preprocesamiento y split
│       ├── model_training_evaluation.py     # Entrenamiento, selección y evaluación
│       ├── model_monitoring.py              # Detección de data drift
│       ├── app_streamlit.py                 # Dashboard de monitoreo
│       ├── model_deploy.py                  # API del modelo (FastAPI)
│       ├── modelo_scoring.joblib            # Modelo final entrenado
│       └── comparacion_modelos_cv.png       # Gráfico comparativo de modelos
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
└── readme.md
```

## 3. Flujo de versionado

El repositorio tiene tres ramas: `developer`, `certification` y `master`. El trabajo se desarrolla en `developer`, se valida en `certification` y se libera en `master` mediante Pull Requests, con un tag por versión.

| Versión | Contenido |
|---|---|
| v1.0.0 | Estructura inicial de carpetas |
| v1.0.1 | Carga de datos y EDA |
| v1.1.0 | Ingeniería de características y comparación de modelos |
| v1.2.0 | Monitoreo de data drift, app Streamlit, mejora del modelo y README |

## 4. Datos y hallazgos del EDA

Dataset: **10.763 créditos, 23 variables**. Variable objetivo: `Pago_atiempo` (1 = paga, 0 = no paga).

**Hallazgo principal: data leakage en `puntaje`.** El 87% de los registros tiene exactamente el mismo valor (95.227787). En el 13% restante, la variable separa perfectamente las dos clases: todos los que no pagan están por debajo de 63 y todos los que pagan, por encima. Su correlación con el objetivo es de 0,88. Esto indica que el puntaje se calcula **a partir del resultado de pago**, por lo que no estaría disponible al momento de otorgar el crédito. **Se excluye del modelo.** Incluirlo habría dado métricas casi perfectas en evaluación y un modelo inútil en producción.

**Problemas de calidad de datos corregidos:**

- `tendencia_ingresos`: 58 valores corruptos convertidos a nulos e imputados.
- `edad_cliente` y `salario_cliente`: 150 registros con edades imposibles (121 a 123 años) y salarios inflados, tratados como errores de carga.
- `salario_cliente`: 5 valores atípicos adicionales por encima de $100M.

**Desbalance de clases:** 95% paga / 5% no paga. Condiciona todas las decisiones de modelado posteriores: split estratificado, pesos de clase y métricas elegidas.

## 5. Ingeniería de características

`ft_engineering.py` concentra la limpieza del EDA en código reproducible (sin depender del notebook) y arma un `ColumnTransformer`:

| Tipo de variable | Tratamiento |
|---|---|
| Numéricas | Imputación por mediana |
| Categóricas nominales | Imputación por moda + OneHotEncoder |
| Categórica ordinal (`tendencia_ingresos`) | Imputación por moda + OrdinalEncoder |

El split train/test es **estratificado** (80/20) para mantener la proporción de morosos en ambos conjuntos. El preprocesador se ajusta solo con los datos de train, dentro del pipeline de cada modelo, para evitar filtración de información desde test.

## 6. Modelado y selección

### Primera versión (v1.1.0) y su problema

Se compararon cinco modelos con SMOTE y se eligió **Logistic Regression** por tener el mayor Recall de "no paga" (0,48) con el umbral por defecto de 0,5.

Al revisar los resultados se encontró un error de criterio: **el recall depende del umbral**, así que comparar recalls a 0,5 entre modelos distintos no mide cuál discrimina mejor. Solo muestra cuál tiene las probabilidades más corridas hacia la clase "no paga". En la práctica, ese modelo marcaba como riesgosos a **~910 de 2.153 clientes** para detectar 49 morosos, apenas mejor que elegir al azar (ROC-AUC 0,57).

### Versión mejorada (v1.2.0)

1. **Selección por PR-AUC** de la clase "no paga", con validación cruzada estratificada de 5 folds sobre train. PR-AUC mide qué tan bien el modelo *ordena* a los clientes por riesgo, sin depender de un umbral, y se concentra en la clase minoritaria.
2. **Pesos de clase en lugar de SMOTE.** SMOTE interpola sobre variables one-hot y genera clientes sintéticos poco realistas.
3. **Escalado** de variables para Logistic Regression (en la v1 entraban sin escalar).
4. **Optimización de hiperparámetros** (RandomizedSearchCV) de los dos mejores modelos.
5. **Umbral de decisión con criterio de negocio**: se fija para detectar al menos el 60% de los morosos, usando predicciones *out-of-fold* de train. El conjunto de test se usa una sola vez, para la evaluación final.

**Comparación por validación cruzada (train):**

| Modelo | PR-AUC | ROC-AUC |
|---|---|---|
| **Random Forest** | **0,156** | **0,691** |
| XGBoost | 0,137 | 0,671 |
| HistGradientBoosting | 0,134 | 0,659 |
| Logistic Regression | 0,117 | 0,661 |
| Decision Tree | 0,088 | 0,636 |

Un modelo al azar tendría un PR-AUC de ~0,048 (la tasa de morosos). La optimización de hiperparámetros no mejoró al Random Forest (0,155 frente a 0,156 por defecto), lo que indica que el límite lo pone la información disponible en los datos y no la configuración del modelo.

**Modelo seleccionado: Random Forest** (`max_depth=12`, `min_samples_leaf=8`, `max_features='sqrt'`, pesos de clase balanceados), umbral 0,2526.

**Evaluación final en test (2.153 clientes, 102 morosos):**

| Métrica | v1 (Logistic Regression) | v2 (Random Forest) |
|---|---|---|
| ROC-AUC | 0,570 | **0,674** |
| PR-AUC | – | **0,132** |
| Recall "no paga" | 0,48 | **0,63** |
| Precision "no paga" | 0,054 | **0,075** |
| Lift sobre el azar | ~1,15 | **1,59** |
| Morosos detectados | 49 / 102 | **64 / 102** |
| Buenos clientes rechazados | 861 | **788** |
| Clientes a marcar para detectar ~48% de morosos | ~910 | **533** |

La v2 detecta **más morosos rechazando menos buenos clientes**.

### Limitaciones y uso recomendado

La señal de los datos es débil (ROC-AUC ~0,67): de cada 13 solicitudes que el modelo marca, 1 corresponde a un moroso. Por eso **no se recomienda usarlo para rechazar automáticamente**, sino para **priorizar solicitudes a revisión manual**. Con ese uso, el equipo de riesgo concentra su análisis en el 40% de las solicitudes donde está el 63% de los morosos.

El umbral es un parámetro de negocio (`RECALL_OBJETIVO` en `model_training_evaluation.py`). Si el costo de un impago sube frente al de perder un cliente, se baja el umbral, y viceversa.

## 7. Monitoreo de data drift

`model_monitoring.py` compara la población **histórica** (70% de los créditos más antiguos según `fecha_prestamo`) con la población **actual** (30% más reciente), simulando el monitoreo periódico en producción.

| Tipo de variable | Métricas |
|---|---|
| Numéricas | Kolmogorov-Smirnov, Population Stability Index (PSI), Jensen-Shannon |
| Categóricas | Chi-cuadrado |

Umbrales de alerta: PSI > 0,10 moderado y > 0,25 crítico; p-value < 0,05 en KS y Chi-cuadrado.

**Resultado:** se detectó **drift crítico en `plazo_meses` y `promedio_ingresos_datacredito`**. Cambiaron el plazo de los créditos otorgados y el perfil de ingresos de los solicitantes respecto de los datos con los que se entrenó el modelo. Se recomienda monitorear el desempeño real del modelo en la población reciente y reentrenarlo si las métricas caen.

### Dashboard en Streamlit

`app_streamlit.py` muestra una tabla con semáforo por variable y los histogramas comparativos histórico vs. actual.

## 8. Cómo ejecutar el proyecto

Requiere **Python 3.13**. Las versiones de las librerías están fijadas en `requirements.txt` porque el modelo guardado (`.joblib`) depende de ellas.

```bash
# 1. Clonar el repositorio
git clone https://github.com/Sirelois/mlops-scoring-crediticio.git
cd mlops-scoring-crediticio

# 2. Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar y evaluar los modelos (genera modelo_scoring.joblib)
python mlops_pipeline/src/model_training_evaluation.py

# 5. Levantar el dashboard de monitoreo
streamlit run mlops_pipeline/src/app_streamlit.py
```

El dashboard queda disponible en `http://localhost:8501`.

## 9. Stack tecnológico

Python · pandas · numpy · scikit-learn · XGBoost · SciPy · matplotlib · seaborn · Streamlit · FastAPI · Git/GitHub
