#!/usr/bin/env bash
# Continuous-integration gate, run before every commit/merge (or as a Git pre-push hook):
#   1. code style (flake8)   2. unit/integration tests + test report
#   3. DVC pipeline: re-runs only the stages whose code, data or params changed
#   4. the evaluate stage's acceptance gate fails the run if the model is not good enough
#   5. show what changed in metrics compared with the last commit
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== 1/4 lint";        python -m flake8 src backend tests scripts
echo "== 2/4 tests";       python scripts/run_tests.py
echo "== 3/4 dvc repro";   dvc repro
echo "== 4/4 metrics diff vs HEAD"; dvc metrics diff HEAD --md || true
echo "CI passed. Commit with: git add dvc.lock reports docs/test_report.md && git commit -m '...' && dvc push"
