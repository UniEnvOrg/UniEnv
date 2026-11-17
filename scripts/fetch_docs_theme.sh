#!/usr/bin/env bash
# Fetch the shared UniEnv docs theme (docs-theme/ of UniEnvOrg.github.io) into .org-site/,
# where mkdocs.yml's theme.custom_dir expects it. CI does the same with actions/checkout.
#
#   scripts/fetch_docs_theme.sh            # from GitHub
#   scripts/fetch_docs_theme.sh ../UniEnvOrg.github.io   # from a local checkout
set -euo pipefail
cd "$(dirname "$0")/.."
src="${1:-https://github.com/UniEnvOrg/UniEnvOrg.github.io.git}"
rm -rf .org-site
if [ -d "$src" ]; then
  mkdir -p .org-site && cp -R "$src/docs-theme" .org-site/docs-theme
else
  git clone --quiet --depth 1 --filter=blob:none --sparse "$src" .org-site
  git -C .org-site sparse-checkout set docs-theme
fi
echo "docs theme: .org-site/docs-theme"
