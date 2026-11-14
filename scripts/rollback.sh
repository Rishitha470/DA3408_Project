#!/usr/bin/env bash
# Roll the served model back to the version produced by an earlier Git commit.
#   ./scripts/rollback.sh <git-commit>        e.g. ./scripts/rollback.sh HEAD~1
# DVC restores models/ exactly as it was at that commit (from the DVC cache), then the model
# server and backend are restarted so they serve it. Nothing is retrained.
set -euo pipefail
COMMIT="${1:?usage: rollback.sh <git-commit>}"
git checkout "$COMMIT" -- dvc.lock
dvc checkout models
echo "models/ restored to $COMMIT: $(python -c "import json;print(json.load(open('models/model_meta.json'))['model_version'])")"
if docker compose ps --status running 2>/dev/null | grep -q model-server; then
  docker compose restart model-server backend
  echo "model-server and backend restarted"
fi
echo "Commit the rollback with: git commit -am 'rollback model to $COMMIT'"
