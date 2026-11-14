#!/usr/bin/env bash
# Roll the served model back to the version produced by an earlier Git commit.
#   ./scripts/rollback.sh <git-commit>        e.g. ./scripts/rollback.sh HEAD~1
# 1. DVC restores models/ exactly as it was at that commit (from the DVC cache); nothing is retrained.
# 2. The evaluation reports of that commit are restored, so the app's pipeline console matches the model.
# 3. The MLflow registry alias 'champion' is moved back to the registry version of that commit
#    (needs MLFLOW_TRACKING_URI, e.g. http://localhost:5000).
# 4. The model server and backend are restarted so they serve it.
set -euo pipefail
COMMIT="${1:?usage: rollback.sh <git-commit>}"
cd "$(dirname "$0")/.."

git checkout "$COMMIT" -- dvc.lock reports
dvc checkout models
echo "models/ restored to $COMMIT: $(python -c "import json;print(json.load(open('models/model_meta.json'))['model_version'])")"

VERSION=$(python -c "import json;print(json.load(open('reports/metrics.json'))['registry']['version'])" 2>/dev/null || true)
NAME=$(python -c "import yaml;print(yaml.safe_load(open('params.yaml'))['evaluate']['registered_model_name'])")
if [ -n "$VERSION" ] && [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
  python -c "import sys, mlflow; mlflow.MlflowClient().set_registered_model_alias(sys.argv[1], 'champion', sys.argv[2])" \
    "$NAME" "$VERSION"
  echo "registry: $NAME@champion -> version $VERSION"
else
  echo "registry alias not changed (MLFLOW_TRACKING_URI not set or no registry version in reports/metrics.json)"
fi

if docker compose ps --status running 2>/dev/null | grep -q model-server; then
  docker compose restart model-server backend
  echo "model-server and backend restarted"
fi
echo "Commit the rollback with: git commit -am 'Rollback model to $COMMIT'"
