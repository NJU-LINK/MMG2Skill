#!/usr/bin/env bash
# Download RLCard opponent ("人机") weights from a GitHub Release into
# rlcard_models/. This restores the .pt checkpoints that are gitignored and
# therefore absent after a fresh clone.
#
# It downloads every <game>_<agent>.tar.gz asset on the release and extracts
# each one back into rlcard_models/<game>_<agent>/.
#
# Prereqs: `gh auth login` (read access is enough for public repos).
#
# Usage:
#     bash scripts/rlcard/download_weights.sh                 # tag weights-v1
#     bash scripts/rlcard/download_weights.sh --tag weights-v2
#     bash scripts/rlcard/download_weights.sh --repo OWNER/REPO
#     bash scripts/rlcard/download_weights.sh --game doudizhu_dmc  # one folder
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

TAG="weights-v1"
REPO=""
ONLY="" # restrict to a single asset basename, e.g. doudizhu_dmc
MODELS_ROOT="rlcard_models"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download <game>_<agent>.tar.gz assets from a GitHub Release and extract them
into rlcard_models/<game>_<agent>/.

Options:
  --tag TAG       Release tag to pull from (default: $TAG)
  --repo SLUG     Source repo OWNER/REPO (default: inferred from origin)
  --game NAME     Download only the NAME.tar.gz asset (default: all)
  -h, --help      Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)  TAG="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --game) ONLY="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

command -v gh >/dev/null 2>&1 || { echo "Error: gh (GitHub CLI) not found. Install: https://cli.github.com" >&2; exit 1; }

REPO_ARGS=()
if [ -n "$REPO" ]; then
    REPO_ARGS=(--repo "$REPO")
fi

if ! gh release view "$TAG" "${REPO_ARGS[@]}" >/dev/null 2>&1; then
    echo "Error: release '$TAG' not found on ${REPO:-<origin>}." >&2
    echo "List available releases: gh release list ${REPO:+--repo $REPO}" >&2
    exit 1
fi

PATTERN="*.tar.gz"
if [ -n "$ONLY" ]; then
    PATTERN="${ONLY}.tar.gz"
fi

mkdir -p "$MODELS_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Repo:    ${REPO:-<origin>}"
echo "Tag:     $TAG"
echo "Pattern: $PATTERN"
echo ""

echo "Downloading assets..."
gh release download "$TAG" "${REPO_ARGS[@]}" --pattern "$PATTERN" --dir "$TMP" --clobber

shopt -s nullglob
archives=("$TMP"/*.tar.gz)
if [ "${#archives[@]}" -eq 0 ]; then
    echo "Error: no assets matched '$PATTERN' on release $TAG." >&2
    exit 1
fi

for archive in "${archives[@]}"; do
    echo "━━━ Extracting $(basename "$archive") ━━━"
    # Tarballs store entries as <game>_<agent>/checkpoint_*.pt
    tar xzf "$archive" -C "$MODELS_ROOT"
done

echo ""
echo "Done. Weights restored under $MODELS_ROOT/:"
for d in "$MODELS_ROOT"/*/; do
    [ -d "$d" ] || continue
    if ls "$d"/*.pt >/dev/null 2>&1; then
        count="$(ls "$d"/*.pt | wc -l | tr -d ' ')"
        echo "  $(basename "$d")  ($count checkpoint file(s))"
    fi
done
