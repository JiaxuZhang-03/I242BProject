#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-"$REPO_ROOT/scratch/huggingface_space"}"
CHECKPOINT="${CHECKPOINT_PATH:-"$REPO_ROOT/result/classifier/resnet18_unfrozen_from_frozen_lr1e-4_128_e8/best_model.pt"}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/static" "$OUT_DIR/src/food_project" "$OUT_DIR/model"

cp "$REPO_ROOT/interactive_demo/app.py" "$OUT_DIR/app.py"
cp -R "$REPO_ROOT/interactive_demo/static/." "$OUT_DIR/static/"
cp -R "$REPO_ROOT/src/food_project/." "$OUT_DIR/src/food_project/"
cp "$CHECKPOINT" "$OUT_DIR/model/best_model.pt"
cp "$CHECKPOINT" "$OUT_DIR/best_model.pt"
cp "$REPO_ROOT/interactive_demo/huggingface_space/Dockerfile" "$OUT_DIR/Dockerfile"
cp "$REPO_ROOT/interactive_demo/huggingface_space/README.md" "$OUT_DIR/README.md"
cp "$REPO_ROOT/interactive_demo/huggingface_space/requirements.txt" "$OUT_DIR/requirements.txt"
cp "$REPO_ROOT/interactive_demo/huggingface_space/.gitattributes" "$OUT_DIR/.gitattributes"

cat <<EOF
Hugging Face Space bundle ready:
  $OUT_DIR

Next:
  1. Create a Hugging Face Space with SDK = Docker.
  2. Upload or push the contents of this folder.
  3. Make sure model/best_model.pt is included.
EOF
