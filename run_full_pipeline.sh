#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Run the full food image classification pipeline from data to final summaries.

Usage:
  ./run_full_pipeline.sh

Optional environment variables:
  PYTHON_BIN              Python executable to use.
                          Default: .venv/bin/python if it exists, otherwise python
  IMAGE_SIZE              Image size for training/evaluation. Default: 128
  CLASSIFIER_BATCH_SIZE   Batch size for classifier training. Default: 32
  SUPCON_BATCH_SIZE       Batch size for SupCon pretraining. Default: 64
  EVAL_BATCH_SIZE         Batch size for evaluation. Default: 64
  CLASSIFIER_EPOCHS       Epochs for classifier training. Default: 5
  FINETUNE_EPOCHS         Epochs for full ResNet18 fine-tuning. Default: 8
  FINETUNE_LR             Learning rate for full ResNet18 fine-tuning. Default: 3e-5
  SUPCON_EPOCHS           Epochs for SupCon pretraining. Default: 5
  NUM_WORKERS             DataLoader workers. Default: 0
  GRADCAM_IMAGES          Number of Grad-CAM examples. Default: 12

Example:
  PYTHON_BIN=/path/to/python CLASSIFIER_EPOCHS=10 SUPCON_EPOCHS=20 ./run_full_pipeline.sh
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
SUPCON_BATCH_SIZE="${SUPCON_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-8}"
FINETUNE_LR="${FINETUNE_LR:-3e-5}"
SUPCON_EPOCHS="${SUPCON_EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRADCAM_IMAGES="${GRADCAM_IMAGES:-12}"

SIMPLE_RUN="simple_cnn_${IMAGE_SIZE}_e${CLASSIFIER_EPOCHS}"
RESNET_RUN="resnet18_pretrained_frozen_${IMAGE_SIZE}_e${CLASSIFIER_EPOCHS}"
RESNET_FT_RUN="resnet18_unfrozen_from_frozen_lr${FINETUNE_LR}_${IMAGE_SIZE}_e${FINETUNE_EPOCHS}"
SUPCON_RUN="simple_cnn_supcon_${IMAGE_SIZE}_e${SUPCON_EPOCHS}"
SUPCON_FT_RUN="simple_cnn_supcon_finetune_${IMAGE_SIZE}_e${CLASSIFIER_EPOCHS}"

run_step() {
  local title="$1"
  shift
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${title}"
  "$@"
}

printf 'Food classification pipeline\n'
printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Python: %s\n' "${PYTHON_BIN}"
printf 'Image size: %s\n' "${IMAGE_SIZE}"
printf 'Classifier epochs: %s\n' "${CLASSIFIER_EPOCHS}"
printf 'ResNet18 fine-tune epochs: %s\n' "${FINETUNE_EPOCHS}"
printf 'ResNet18 fine-tune LR: %s\n' "${FINETUNE_LR}"
printf 'SupCon epochs: %s\n' "${SUPCON_EPOCHS}"
printf 'Num workers: %s\n' "${NUM_WORKERS}"

run_step "Check Python environment" \
  "${PYTHON_BIN}" -c "import torch, torchvision, pandas, sklearn, matplotlib; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"

run_step "Generate EDA outputs" \
  "${PYTHON_BIN}" src/run_eda.py \
    --output-dir result/eda \
    --samples-per-class 6

run_step "Train simple CNN baseline" \
  "${PYTHON_BIN}" src/train_classifier.py \
    --model-name simple_cnn \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --patience 3 \
    --experiment-name "${SIMPLE_RUN}"

run_step "Train pretrained frozen ResNet18" \
  "${PYTHON_BIN}" src/train_classifier.py \
    --model-name resnet18 \
    --pretrained \
    --freeze-backbone \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --patience 3 \
    --experiment-name "${RESNET_RUN}"

run_step "Fine-tune full ResNet18 from frozen checkpoint" \
  "${PYTHON_BIN}" src/train_classifier.py \
    --model-name resnet18 \
    --init-checkpoint "result/classifier/${RESNET_RUN}/best_model.pt" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --epochs "${FINETUNE_EPOCHS}" \
    --lr "${FINETUNE_LR}" \
    --num-workers "${NUM_WORKERS}" \
    --patience 4 \
    --experiment-name "${RESNET_FT_RUN}"

run_step "Run supervised contrastive pretraining" \
  "${PYTHON_BIN}" src/pretrain_supcon.py \
    --model-name simple_cnn \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${SUPCON_BATCH_SIZE}" \
    --epochs "${SUPCON_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --experiment-name "${SUPCON_RUN}"

run_step "Fine-tune classifier from SupCon encoder" \
  "${PYTHON_BIN}" src/train_classifier.py \
    --model-name simple_cnn \
    --supcon-checkpoint "result/supcon/${SUPCON_RUN}/best_supcon.pt" \
    --image-size "${IMAGE_SIZE}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --patience 3 \
    --experiment-name "${SUPCON_FT_RUN}"

run_step "Evaluate simple CNN baseline" \
  "${PYTHON_BIN}" src/evaluate_model.py \
    --checkpoint "result/classifier/${SIMPLE_RUN}/best_model.pt" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "result/evaluation/${SIMPLE_RUN}"

run_step "Evaluate pretrained frozen ResNet18" \
  "${PYTHON_BIN}" src/evaluate_model.py \
    --checkpoint "result/classifier/${RESNET_RUN}/best_model.pt" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "result/evaluation/${RESNET_RUN}"

run_step "Evaluate fine-tuned ResNet18" \
  "${PYTHON_BIN}" src/evaluate_model.py \
    --checkpoint "result/classifier/${RESNET_FT_RUN}/best_model.pt" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "result/evaluation/${RESNET_FT_RUN}"

run_step "Evaluate SupCon fine-tuned classifier" \
  "${PYTHON_BIN}" src/evaluate_model.py \
    --checkpoint "result/classifier/${SUPCON_FT_RUN}/best_model.pt" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "result/evaluation/${SUPCON_FT_RUN}"

run_step "Generate Grad-CAM examples for the strongest default model" \
  "${PYTHON_BIN}" src/make_gradcam.py \
    --checkpoint "result/classifier/${RESNET_FT_RUN}/best_model.pt" \
    --split test \
    --num-images "${GRADCAM_IMAGES}" \
    --output-dir "result/gradcam/${RESNET_FT_RUN}"

run_step "Build experiment summary tables" \
  "${PYTHON_BIN}" src/summarize_results.py

run_step "Build final summary plots" \
  "${PYTHON_BIN}" result/make_summary_plots.py

printf '\nPipeline complete.\n'
printf 'Summary table: result/experiment_summary.csv\n'
printf 'Summary plots: result/summary_plots/\n'
