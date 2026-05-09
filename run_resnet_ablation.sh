#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Run the focused ResNet18 fine-tuning ablation study.

Usage:
  ./run_resnet_ablation.sh

Optional environment variables:
  PYTHON_BIN              Python executable to use.
                          Default: .venv/bin/python if it exists, otherwise python
  IMAGE_SIZE              Image size for training/evaluation. Default: 128
  CLASSIFIER_BATCH_SIZE   Batch size for training. Default: 32
  EVAL_BATCH_SIZE         Batch size for evaluation. Default: 64
  CLASSIFIER_EPOCHS       Epochs for the frozen ResNet18 seed model. Default: 5
  FINETUNE_EPOCHS         Epochs for ablation fine-tuning runs. Default: 8
  LAYER4_LR               Learning rate for layer4-only fine-tuning. Default: 3e-5
  FULL_LRS                Space-separated full fine-tuning learning rates.
                          Default: "1e-5 3e-5 1e-4"
  NUM_WORKERS             DataLoader workers. Default: 0

Example:
  FULL_LRS="3e-5 1e-4" ./run_resnet_ablation.sh
USAGE
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

IMAGE_SIZE="${IMAGE_SIZE:-128}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-8}"
LAYER4_LR="${LAYER4_LR:-3e-5}"
FULL_LRS="${FULL_LRS:-1e-5 3e-5 1e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

FROZEN_RUN="resnet18_pretrained_frozen_${IMAGE_SIZE}_e${CLASSIFIER_EPOCHS}"
LAYER4_RUN="resnet18_layer4_from_frozen_lr${LAYER4_LR}_${IMAGE_SIZE}_e${FINETUNE_EPOCHS}"

run_step() {
  local title="$1"
  shift
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${title}"
  "$@"
}

printf 'ResNet18 ablation study\n'
printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Python: %s\n' "${PYTHON_BIN}"
printf 'Full fine-tuning LRs: %s\n' "${FULL_LRS}"

if [[ ! -f "result/classifier/${FROZEN_RUN}/best_model.pt" ]]; then
  run_step "Train frozen ResNet18 seed model" \
    "${PYTHON_BIN}" src/train_classifier.py \
      --model-name resnet18 \
      --pretrained \
      --freeze-backbone \
      --image-size "${IMAGE_SIZE}" \
      --batch-size "${CLASSIFIER_BATCH_SIZE}" \
      --epochs "${CLASSIFIER_EPOCHS}" \
      --num-workers "${NUM_WORKERS}" \
      --patience 3 \
      --experiment-name "${FROZEN_RUN}"
fi

run_step "Fine-tune ResNet18 layer4 only" \
  "${PYTHON_BIN}" src/train_classifier.py \
    --model-name resnet18 \
    --init-checkpoint "result/classifier/${FROZEN_RUN}/best_model.pt" \
    --trainable-backbone resnet-layer4 \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --lr "${LAYER4_LR}" \
    --num-workers "${NUM_WORKERS}" \
    --patience 4 \
    --experiment-name "${LAYER4_RUN}"

run_step "Evaluate layer4-only fine-tuning" \
  "${PYTHON_BIN}" src/evaluate_model.py \
    --checkpoint "result/classifier/${LAYER4_RUN}/best_model.pt" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "result/evaluation/${LAYER4_RUN}"

for LR in ${FULL_LRS}; do
  FULL_RUN="resnet18_unfrozen_from_frozen_lr${LR}_${IMAGE_SIZE}_e${FINETUNE_EPOCHS}"
  run_step "Fine-tune full ResNet18 with LR=${LR}" \
    "${PYTHON_BIN}" src/train_classifier.py \
      --model-name resnet18 \
      --init-checkpoint "result/classifier/${FROZEN_RUN}/best_model.pt" \
      --image-size "${IMAGE_SIZE}" \
      --batch-size "${CLASSIFIER_BATCH_SIZE}" \
      --epochs "${FINETUNE_EPOCHS}" \
      --lr "${LR}" \
      --num-workers "${NUM_WORKERS}" \
      --patience 4 \
      --experiment-name "${FULL_RUN}"

  run_step "Evaluate full ResNet18 LR=${LR}" \
    "${PYTHON_BIN}" src/evaluate_model.py \
      --checkpoint "result/classifier/${FULL_RUN}/best_model.pt" \
      --split test \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --output-dir "result/evaluation/${FULL_RUN}"
done

run_step "Rebuild experiment summary" \
  "${PYTHON_BIN}" src/summarize_results.py

run_step "Build ablation summary" \
  "${PYTHON_BIN}" src/summarize_ablation.py

printf '\nAblation complete.\n'
printf 'Ablation table: result/ablation_summary.csv\n'
printf 'Ablation plots: result/summary_plots/resnet18_ablation_*.png\n'
