#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/dist"
mojo build --emit shared-lib "$repo_root/src/capi.mojo" \
  -o "$repo_root/dist/libmojo-prophet.so"
