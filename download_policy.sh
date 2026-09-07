#!/usr/bin/env bash
# Fetches the pretrained policy checkpoint (body_latest.jit,
# adaptation_module_latest.jit) from the GitHub release of this repo and
# verifies each file's SHA-256 hash against the values recorded in AUDIT.md.
#
# Usage:
#   ./download_policy.sh
#
# Env overrides:
#   GO2_POLICY_DIR   destination directory (default: repo-relative
#                    stage2-go2-mujoco-inference/checkpoints, matching the
#                    default POLICY_DIR in harness.py and the inference
#                    scripts)
#   POLICY_RELEASE_TAG   release tag to fetch from (default: v1.0-policy)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${GO2_POLICY_DIR:-$REPO_ROOT/stage2-go2-mujoco-inference/checkpoints}"
TAG="${POLICY_RELEASE_TAG:-v1.0-policy}"
BASE_URL="https://github.com/Ashfak-Kausik/rl-locomotion-learning/releases/download/${TAG}"

declare -A EXPECTED_SHA256=(
  ["body_latest.jit"]="7b6e604e2147742a89ef50d91e7ee501023331b2589d1c3143a9d2ba858db7b5"
  ["adaptation_module_latest.jit"]="0e091f829dcfbedd4ccca6752863e1e2feca105f79da07d04e3545b8815dcc13"
)

mkdir -p "$DEST_DIR"

echo "Downloading policy checkpoint (tag: $TAG) into: $DEST_DIR"

for fname in "${!EXPECTED_SHA256[@]}"; do
  url="${BASE_URL}/${fname}"
  dest="${DEST_DIR}/${fname}"
  echo "  fetching ${fname} ..."
  curl -fL --retry 3 -o "$dest" "$url"

  expected="${EXPECTED_SHA256[$fname]}"
  actual="$(sha256sum "$dest" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: SHA-256 mismatch for ${fname}" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    rm -f "$dest"
    exit 1
  fi
  echo "  OK: ${fname} (sha256 verified)"
done

echo "Done. Checkpoint files verified and installed in $DEST_DIR"
echo "harness.py and the inference scripts will pick this up automatically"
echo "(override with GO2_POLICY_DIR if you placed the files elsewhere)."
