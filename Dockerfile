# Imagen de la API de Scoring Crediticio
# Misma versión de Python con la que se entrenó el modelo
FROM python:3.13-slim

WORKDIR /app

# Dependencias primero: Docker cachea esta capa si no cambian
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código de la API y modelo entrenado
COPY mlops_pipeline/src/model_deploy.py mlops_pipeline/src/modelo_scoring.joblib ./mlops_pipeline/src/

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline/src", "--host", "0.0.0.0", "--port", "8000"]
