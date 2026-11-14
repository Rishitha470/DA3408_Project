#!/usr/bin/env bash
# Installs a Git pre-push hook that
#   1. runs the CI gate (scripts/ci.sh) and blocks the push if it fails, then
#   2. runs Git LFS's own pre-push step, which uploads the LFS files (screenshots).
# Git LFS installs its own pre-push hook; replacing it with only the CI gate would push the
# LFS pointer files without the real images, so both steps are kept here.
set -euo pipefail
cd "$(dirname "$0")/.."
cat > .git/hooks/pre-push <<'EOF'
#!/usr/bin/env bash
refs=$(cat)                      # Git passes the refs to push on stdin; Git LFS needs them too
./scripts/ci.sh || exit 1
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "git-lfs is not installed: cannot upload LFS files"; exit 2; }
printf '%s\n' "$refs" | git lfs pre-push "$@"
EOF
chmod +x .git/hooks/pre-push
echo "pre-push hook installed (CI gate + Git LFS upload)"
