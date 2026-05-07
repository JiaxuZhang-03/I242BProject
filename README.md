# INDENG 1/242B Spring 2026 Project

This repository contains a food image classification pipeline for the INDENG 1/242B Spring 2026 final project. The project compares a simple CNN baseline, an ImageNet-pretrained ResNet18 transfer learning model, and a supervised contrastive learning workflow. It also includes scripts for EDA, model evaluation, Grad-CAM visual explanations, and report-ready summary plots.

## Dataset

The dataset is expected to use an ImageFolder-style split:

```text
data/
  .../
    train/
      healthy/
      unhealthy/
    val/
      healthy/
      unhealthy/
    test/
      healthy/
      unhealthy/
```

The scripts automatically search under `data/` for a directory containing `train/`, `val/`, and `test/`. The `data/` directory is intentionally ignored by Git, except for placeholder files.

## Project layout

- `data/`: local image data, ignored by Git except placeholder files.
- `result/`: generated outputs. Most experiment folders are ignored by Git; `summary_plots/` and top-level summary files are kept for reporting.
- `src/food_project/`: shared dataset, model, training, metrics, plotting, checkpointing, and Grad-CAM logic.
- `src/run_eda.py`: dataset summaries and sample visualizations.
- `src/train_classifier.py`: simple CNN, pretrained backbone, or SupCon-initialized classifier training.
- `src/pretrain_supcon.py`: supervised contrastive pretraining.
- `src/evaluate_model.py`: checkpoint evaluation on train/val/test splits.
- `src/make_gradcam.py`: Grad-CAM visual explanations.
- `src/summarize_results.py`: combines evaluation JSON files and training histories into summary tables.
- `result/make_summary_plots.py`: creates final report-ready plots from generated summaries.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

During development, the project was run in the `nnenv2` environment.

## Reproducing the pipeline

Run EDA:

```bash
python src/run_eda.py
```

Train the simple CNN baseline:

```bash
python src/train_classifier.py \
  --model-name simple_cnn \
  --image-size 128 \
  --batch-size 32 \
  --epochs 5 \
  --num-workers 0 \
  --experiment-name simple_cnn_128_e5
```

Train an ImageNet-pretrained ResNet18 transfer learning model:

```bash
python src/train_classifier.py \
  --model-name resnet18 \
  --pretrained \
  --freeze-backbone \
  --image-size 128 \
  --batch-size 32 \
  --epochs 5 \
  --num-workers 0 \
  --experiment-name resnet18_pretrained_frozen_128_e5
```

Run supervised contrastive pretraining:

```bash
python src/pretrain_supcon.py \
  --model-name simple_cnn \
  --image-size 128 \
  --batch-size 64 \
  --epochs 5 \
  --num-workers 0 \
  --experiment-name simple_cnn_supcon_128_e5
```

Fine-tune a classifier from the SupCon encoder:

```bash
python src/train_classifier.py \
  --model-name simple_cnn \
  --supcon-checkpoint result/supcon/simple_cnn_supcon_128_e5/best_supcon.pt \
  --image-size 128 \
  --batch-size 32 \
  --epochs 5 \
  --num-workers 0 \
  --experiment-name simple_cnn_supcon_finetune_128_e5
```

Evaluate a trained checkpoint:

```bash
python src/evaluate_model.py \
  --checkpoint result/classifier/<run>/best_model.pt \
  --split test \
  --batch-size 64 \
  --num-workers 0
```

Generate Grad-CAM visualizations:

```bash
python src/make_gradcam.py \
  --checkpoint result/classifier/resnet18_pretrained_frozen_128_e5/best_model.pt \
  --split test \
  --num-images 12
```

Summarize results and generate report plots:

```bash
python src/summarize_results.py
python result/make_summary_plots.py
```

## Current experiment summary

The latest summary is saved in:

- `result/experiment_summary.csv`
- `result/experiment_summary.md`
- `result/summary_plots/`

Current test-set results:

| Experiment | Best validation accuracy | Test accuracy | Macro F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 pretrained frozen | 0.8563 | 0.8697 | 0.8697 | 0.9344 |
| Simple CNN + SupCon fine-tune | 0.8084 | 0.8046 | 0.8033 | 0.8813 |
| Simple CNN baseline | 0.7989 | 0.8027 | 0.8027 | 0.8869 |

## Git notes

Large local artifacts are ignored:

- `data/`
- model checkpoints
- detailed experiment folders such as `result/classifier/`, `result/evaluation/`, `result/gradcam/`, and `result/supcon/`

The repository keeps only lightweight summary tables, summary plots, and source code needed to reproduce the pipeline.
