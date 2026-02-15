#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# sync_to_release.sh — Copy code + models from strandweaver-dev → strandweaver
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/sync_to_release.sh                    # dry-run (preview)
#   ./scripts/sync_to_release.sh --apply            # actually sync
#   ./scripts/sync_to_release.sh --apply --commit   # sync + auto-commit
#
# What gets synced:
#   ✓ All Python source code    (strandweaver/)
#   ✓ Tests                     (tests/)
#   ✓ Scripts                   (scripts/)
#   ✓ Notebooks                 (notebooks/)
#   ✓ Trained model weights     (trained_models_10x/ → trained_models/)
#   ✓ Config/docs               (setup.py, README.md, TRAINING.md, etc.)
#
# What does NOT sync:
#   ✗ Training data CSVs        (training_data_10x/)
#   ✗ Tarballs                  (graph_csvs.tar.gz)
#   ✗ Logs                      (training_*.log)
#   ✗ Git history               (.git/)
#   ✗ Python caches             (__pycache__/, *.egg-info)
#
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEV_REPO="$(dirname "$SCRIPT_DIR")"
RELEASE_REPO="${DEV_REPO}/../strandweaver"

# ── Parse args ────────────────────────────────────────────────────────
DRY_RUN=true
AUTO_COMMIT=false
for arg in "$@"; do
    case "$arg" in
        --apply)  DRY_RUN=false ;;
        --commit) AUTO_COMMIT=true ;;
        --help|-h)
            head -30 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────
if [[ ! -d "$DEV_REPO/strandweaver" ]]; then
    echo "ERROR: Cannot find strandweaver-dev at $DEV_REPO"
    exit 1
fi
if [[ ! -d "$RELEASE_REPO/.git" ]]; then
    echo "ERROR: Cannot find strandweaver release repo at $RELEASE_REPO"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        StrandWeaver  ·  Dev → Release Sync                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Source : $DEV_REPO"
echo "║  Target : $RELEASE_REPO"
if $DRY_RUN; then
    echo "║  Mode   : DRY RUN (use --apply to sync)"
else
    echo "║  Mode   : APPLY"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Ensure release repo is on dev branch ──────────────────────────────
CURRENT_BRANCH=$(cd "$RELEASE_REPO" && git branch --show-current)
if [[ "$CURRENT_BRANCH" != "dev" ]]; then
    echo "⚠  Release repo is on branch '$CURRENT_BRANCH', switching to 'dev'..."
    if ! $DRY_RUN; then
        (cd "$RELEASE_REPO" && git checkout dev)
    fi
fi

# ── Rsync flags ───────────────────────────────────────────────────────
RSYNC_OPTS=(
    -av
    --delete
    --exclude='.git/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.egg-info/'
    --exclude='training_data*/'
    --exclude='test_graph_only/'
    --exclude='graph_csvs*.tar.gz'
    --exclude='training_*.log'
    --exclude='.DS_Store'
    --exclude='archive/'
    --exclude='models/'
    --exclude='test_output/'
    --exclude='venv*/'
    --exclude='.nextflow*'
    --exclude='work/'
    --exclude='.env'
)

if $DRY_RUN; then
    RSYNC_OPTS+=(--dry-run)
fi

# ── Sync code ─────────────────────────────────────────────────────────
echo "── Syncing source code & config ──────────────────────────────"
rsync "${RSYNC_OPTS[@]}" \
    "$DEV_REPO/strandweaver/" "$RELEASE_REPO/strandweaver/"
rsync "${RSYNC_OPTS[@]}" \
    "$DEV_REPO/tests/" "$RELEASE_REPO/tests/"
rsync "${RSYNC_OPTS[@]}" \
    "$DEV_REPO/scripts/" "$RELEASE_REPO/scripts/"
rsync "${RSYNC_OPTS[@]}" \
    "$DEV_REPO/notebooks/" "$RELEASE_REPO/notebooks/"

# Top-level config files
for f in setup.py README.md TRAINING.md env.yml Dockerfile LICENSE; do
    if [[ -f "$DEV_REPO/$f" ]]; then
        if $DRY_RUN; then
            echo "  → $f"
        else
            cp "$DEV_REPO/$f" "$RELEASE_REPO/$f"
        fi
    fi
done

# ── Sync trained models ──────────────────────────────────────────────
echo ""
echo "── Syncing trained model weights ─────────────────────────────"
if [[ -d "$DEV_REPO/trained_models_10x" ]]; then
    # Copy 10x models as the primary trained_models in release
    rsync "${RSYNC_OPTS[@]}" \
        "$DEV_REPO/trained_models_10x/" "$RELEASE_REPO/trained_models/"
    echo "  trained_models_10x/ → trained_models/"
fi

# ── Summary ───────────────────────────────────────────────────────────
echo ""
if $DRY_RUN; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "  DRY RUN complete. Run with --apply to execute."
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✓ Sync complete"
    echo "═══════════════════════════════════════════════════════════════"

    # Show what changed in release repo
    echo ""
    echo "── Changes in release repo ───────────────────────────────────"
    (cd "$RELEASE_REPO" && git status --short | head -30)
    CHANGED=$(cd "$RELEASE_REPO" && git status --short | wc -l | tr -d ' ')
    echo "  $CHANGED files changed"

    if $AUTO_COMMIT && [[ "$CHANGED" -gt 0 ]]; then
        echo ""
        echo "── Auto-committing to dev branch ─────────────────────────────"
        (cd "$RELEASE_REPO" && \
            git add -A && \
            git commit -m "sync: update from strandweaver-dev $(date +%Y-%m-%d)")
        echo "  ✓ Committed. Push with: cd $RELEASE_REPO && git push origin dev"
    fi
fi
