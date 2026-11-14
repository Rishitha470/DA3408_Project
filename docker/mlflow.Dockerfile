# One image for both the MLflow tracking server and the MLflow model server.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY docker/requirements-model.txt /tmp/requirements-model.txt
RUN pip install --no-cache-dir -r /tmp/requirements-model.txt
WORKDIR /mlflow
EXPOSE 5000 5001
