#!/usr/bin/env bash
# Installs scripts/ci.sh as a Git pre-push hook, so the CI gate runs automatically before every push.
set -euo pipefail
cd "$(dirname "$0")/.."
printf '#!/usr/bin/env bash\nexec ./scripts/ci.sh\n' > .git/hooks/pre-push
chmod +x .git/hooks/pre-push
echo "pre-push hook installed"
