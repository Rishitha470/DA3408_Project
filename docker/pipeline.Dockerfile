# Optional: run the whole DVC pipeline (Ray + training) inside a container for a reproducible environment.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
COPY requirements.txt /tmp/requirements.txt
COPY docker/requirements-model.txt /tmp/docker/requirements-model.txt
RUN cd /tmp && pip install --no-cache-dir -r requirements.txt
WORKDIR /workspace
CMD ["dvc", "repro"]
