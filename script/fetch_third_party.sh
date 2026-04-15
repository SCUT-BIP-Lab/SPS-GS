#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="$ROOT_DIR/submodules"

mkdir -p "$SUBMODULE_DIR"

fetch_repo() {
  local url="$1"
  local target="$2"
  if [ -d "$target" ]; then
    echo "Already exists: $target"
  else
    git clone "$url" "$target"
  fi
}

fetch_repo "https://github.com/ashawkey/diff-gaussian-rasterization.git" \
  "$SUBMODULE_DIR/diff-gaussian-rasterization"
fetch_repo "https://gitlab.inria.fr/bkerbl/simple-knn.git" \
  "$SUBMODULE_DIR/simple-knn"

echo "Third-party repositories are ready under $SUBMODULE_DIR"
