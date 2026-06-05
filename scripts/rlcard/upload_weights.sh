#!/usr/bin/env bash
# Upload trained RLCard opponent ("人机") weights to a GitHub Release.
#
# Each rlcard_models/<game>_<agent>/ folder is packed into <game>_<agent>.tar.gz
# (only the *.pt checkpoint files) and uploaded as a release asset. Re-running
# overwrites existing assets with the same name.
#
# Prereqs: `gh auth login` with push access to the target repo, and weights
# already present under rlcard_models/ (train them via scripts/rlcard/setup.sh).
#
# Usage:
#     bash scripts/rlcard/upload_weights.sh                 # tag weights-v1, all games
#     bash scripts/rlcard/upload_weights.sh --tag weights-v2
#     bash scripts/rlcard/upload_weights.sh --repo OWNER/REPO
#     bash scripts/rlcard/upload_weights.sh --dir doudizhu_dmc   # only one folder
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

TAG="weights-v1"
REPO=""          # empty => gh infers from the origin remote
ONLY_DIR=""
MODELS_ROOT="rlcard_models"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Pack rlcard_models/<game>_<agent>/*.pt into tarballs and upload them to a
GitHub Release as downloadable assets.

Options:
  --tag TAG       Release tag to create/use (default: $TAG)
  --repo SLUG     Target repo OWNER/REPO (default: inferred from origin)
  --dir NAME      Upload only rlcard_models/NAME (default: every subfolder)
  -h, --help      Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)  TAG="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --dir)  ONLY_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

command -v gh >/dev/null 2>&1 || { echo "Error: gh (GitHub CLI) not found. Install: https://cli.github.com" >&2; exit 1; }

if [ ! -d "$MODELS_ROOT" ]; then
    echo "Error: $MODELS_ROOT/ not found. Train weights first: bash scripts/rlcard/setup.sh" >&2
    exit 1
fi

REPO_ARGS=()
if [ -n "$REPO" ]; then
    REPO_ARGS=(--repo "$REPO")
fi

# Collect target folders.
DIRS=()
if [ -n "$ONLY_DIR" ]; then
    [ -d "$MODELS_ROOT/$ONLY_DIR" ] || { echo "Error: $MODELS_ROOT/$ONLY_DIR not found." >&2; exit 1; }
    DIRS=("$ONLY_DIR")
else
    for d in "$MODELS_ROOT"/*/; do
        [ -d "$d" ] || continue
        name="$(basename "$d")"
        # Skip folders with no checkpoints.
        if ls "$d"/*.pt >/dev/null 2>&1; then
            DIRS+=("$name")
        fi
    done
fi

if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "Error: no rlcard_models/*/ folders with .pt checkpoints found." >&2
    exit 1
fi

echo "Repo:  ${REPO:-<origin>}"
echo "Tag:   $TAG"
echo "Dirs:  ${DIRS[*]}"
echo ""

# Ensure the release exists (create as a draft-free release if missing).
if gh release view "$TAG" "${REPO_ARGS[@]+"${REPO_ARGS[@]}"}" >/dev/null 2>&1; then
    echo "Release $TAG already exists — assets will be overwritten where names collide."
else
    echo "Creating release $TAG..."
    gh release create "$TAG" "${REPO_ARGS[@]+"${REPO_ARGS[@]}"}" \
        --title "RLCard opponent weights ($TAG)" \
        --notes "Trained RLCard opponent (人机) checkpoints. Fetch with scripts/rlcard/download_weights.sh --tag $TAG"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for name in "${DIRS[@]}"; do
    archive="$TMP/${name}.tar.gz"
    echo "━━━ Packing $name ━━━"
    # Store entries as <name>/checkpoint_*.pt so extraction recreates the folder.
    ( cd "$MODELS_ROOT" && tar czf "$archive" "$name"/*.pt )
    size="$(du -h "$archive" | cut -f1)"
    echo "  $(basename "$archive") ($size) → uploading"
    gh release upload "$TAG" "$archive" "${REPO_ARGS[@]+"${REPO_ARGS[@]}"}" --clobber
done

echo ""
echo "Done. Assets uploaded to release $TAG."
echo "Users fetch them with: bash scripts/rlcard/download_weights.sh --tag $TAG"
