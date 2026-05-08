# INDENG 1/242B Spring 2026 Project

## Research Overview

This repository implements a reproducible image classification study for the INDENG 1/242B Spring 2026 final project. The task is binary food image classification: given an image, the model predicts whether the food is `healthy` or `unhealthy`.

The study is organized as an incremental experimental pipeline rather than a single model training script. It begins with exploratory data analysis, establishes a simple convolutional neural network baseline, evaluates transfer learning with an ImageNet-pretrained ResNet18 backbone, improves the transfer model through full fine-tuning, and includes a supervised contrastive learning branch for representation learning. The final pipeline also generates evaluation tables, training curves, confusion matrices, ROC curves, and Grad-CAM visual explanations for model interpretability.

The strongest model in the current study is a fully fine-tuned ResNet18 initialized from a frozen-transfer checkpoint. It achieves `0.9406` test accuracy and `0.9845` ROC AUC on the held-out test set.

## Research Questions

This project investigates three practical questions in food image classification:

1. How much performance can be obtained from a small task-specific CNN trained from scratch?
2. How much does ImageNet transfer learning improve classification performance under the same train/validation/test protocol?
3. Does further representation adaptation, through either full backbone fine-tuning or supervised contrastive pretraining, improve held-out test performance and provide stronger report-ready evidence?

The experimental design treats the simple CNN as a baseline, the frozen ResNet18 as the primary transfer learning comparison, the fully fine-tuned ResNet18 as the main accuracy-improvement experiment, and the SupCon branch as an additional representation-learning comparison.

## Dataset

The dataset is expected to use an ImageFolder-style directory structure:

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

The scripts automatically search under `data/` for a directory containing `train/`, `val/`, and `test/` subdirectories. The local dataset used in the current experiments contains `5,222` images:

| Split | Healthy | Unhealthy | Total |
| --- | ---: | ---: | ---: |
| Train | 2,089 | 2,089 | 4,178 |
| Validation | 261 | 261 | 522 |
| Test | 261 | 261 | 522 |

The class distribution is balanced in every split, which makes accuracy, macro F1, and weighted F1 directly comparable. The `data/` directory is intentionally ignored by Git, except for placeholder files, because image datasets should not be pushed to the repository.

## Methodology

The project uses PyTorch and torchvision. All major logic is placed in the reusable package `src/food_project/`, while top-level scripts provide explicit experiment entry points.

The modeling workflow includes four main experiments:

1. **Simple CNN baseline**
   A compact convolutional network trained from scratch. This model establishes the minimum expected performance from task-specific learning without external pretrained features.

2. **Frozen ImageNet-pretrained ResNet18**
   A ResNet18 backbone initialized with ImageNet weights, with the feature extractor frozen and only the classification head trained. This experiment isolates the value of generic visual representations.

3. **Full ResNet18 fine-tuning**
   The best frozen ResNet18 checkpoint is used to initialize a second training stage. The backbone is then unfrozen and optimized with a small learning rate (`3e-5`). This tests whether adapting pretrained features to the food dataset improves generalization.

4. **Supervised contrastive pretraining and fine-tuning**
   A supervised contrastive loss is used to pretrain an encoder with label-aware representation learning. The encoder is then reused for classifier fine-tuning. This provides an alternative representation-learning path for comparison with ImageNet transfer learning.

For evaluation, each trained classifier is assessed on the held-out test split. The summary pipeline reports accuracy, macro F1, weighted F1, ROC AUC, confusion matrices, and training history plots. Grad-CAM is generated for the strongest model to support qualitative interpretability.

## Current Results

The current test-set results are:

| Experiment | Best validation accuracy | Test accuracy | Macro F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 full fine-tune | 0.9598 | 0.9406 | 0.9406 | 0.9845 |
| ResNet18 pretrained frozen | 0.8563 | 0.8697 | 0.8697 | 0.9344 |
| Simple CNN + SupCon fine-tune | 0.8084 | 0.8046 | 0.8033 | 0.8813 |
| Simple CNN baseline | 0.7989 | 0.8027 | 0.8027 | 0.8869 |

The strongest improvement comes from unfreezing and fine-tuning ResNet18 after the frozen-transfer stage. Compared with the frozen ResNet18 model, full fine-tuning improves test accuracy from `0.8697` to `0.9406`, an absolute increase of approximately `7.09` percentage points. The final model also improves ROC AUC from `0.9344` to `0.9845`.

The final ResNet18 fine-tuned model has the following held-out test confusion matrix:

| True class | Predicted healthy | Predicted unhealthy |
| --- | ---: | ---: |
| Healthy | 248 | 13 |
| Unhealthy | 18 | 243 |

This indicates balanced performance across the two classes: healthy recall is `0.9502`, unhealthy recall is `0.9310`, and both class-level F1 scores are approximately `0.94`.

## Generated Research Artifacts

The pipeline creates the following report-ready artifacts:

- EDA outputs: class counts, split summaries, and sample image grids.
- Training outputs: checkpoints, history CSV files, and per-experiment training curves.
- Evaluation outputs: metrics JSON files, predictions CSV files, classification reports, confusion matrices, and ROC curves.
- Interpretability outputs: Grad-CAM visualizations for the strongest model.
- Summary outputs: an experiment summary table and final plots under `result/summary_plots/`.

The lightweight summary artifacts are kept in Git:

- `result/experiment_summary.csv`
- `result/experiment_summary.md`
- `result/summary_plots/`

Large generated folders such as `result/classifier/`, `result/evaluation/`, `result/gradcam/`, and `result/supcon/` are ignored by Git and can be regenerated by running the pipeline.

## Project Layout

- `data/`: local image data, ignored by Git except placeholder files.
- `result/`: generated outputs. Most experiment folders are ignored by Git; `summary_plots/` and top-level summary files are kept for reporting.
- `src/food_project/`: shared dataset, model, training, metrics, plotting, checkpointing, and Grad-CAM logic.
- `src/run_eda.py`: dataset summaries and sample visualizations.
- `src/train_classifier.py`: simple CNN, pretrained backbone, checkpoint-initialized, or SupCon-initialized classifier training.
- `src/pretrain_supcon.py`: supervised contrastive pretraining.
- `src/evaluate_model.py`: checkpoint evaluation on train/validation/test splits.
- `src/make_gradcam.py`: Grad-CAM visual explanations.
- `src/summarize_results.py`: combines evaluation JSON files and training histories into summary tables.
- `download_data.sh`: downloads and validates the dataset into `data/`.
- `setup_environment.sh`: one-command environment setup using `requirements.txt`.
- `run_full_pipeline.sh`: one-command script that runs the full research pipeline.
- `result/make_summary_plots.py`: creates final report-ready plots from generated summaries.

## Reproducibility

### Local Requirements

A collaborator needs the following local tools:

- `git`
- `python3`
- `curl`
- `unzip`
- `bash`
- internet access for dataset download and, on first use, pretrained ResNet18 weights

The frozen Python package requirements are stored in `requirements.txt`.

### Full Reproduction From a Fresh Clone

From a fresh clone, run the following commands from the repository root:

```bash
./download_data.sh
./setup_environment.sh
./run_full_pipeline.sh
```

The first command downloads the Kaggle dataset zip with `curl`, extracts it into `data/`, and validates that the expected `train/`, `val/`, and `test/` folders exist. The second command creates a local `.venv/` and installs packages from `requirements.txt`. The third command runs the complete experimental workflow from EDA to final summary plots.

If a different direct dataset zip URL is needed:

```bash
DATA_URL=<direct-zip-url> ./download_data.sh
```

If an existing Python or conda environment should be used instead of `.venv/`:

```bash
CREATE_VENV=0 PYTHON_BIN=/path/to/python ./setup_environment.sh
PYTHON_BIN=/path/to/python ./run_full_pipeline.sh
```

During development, the project was run in the `nnenv2` environment.

## Full Pipeline

When the environment is already set up, the complete study can be rerun with:

```bash
./run_full_pipeline.sh
```

The script executes:

1. EDA and sample visualization generation.
2. Simple CNN baseline training.
3. ImageNet-pretrained frozen ResNet18 training.
4. Full ResNet18 fine-tuning from the frozen-transfer checkpoint.
5. Supervised contrastive pretraining.
6. SupCon-initialized classifier fine-tuning.
7. Test-set evaluation for all trained classifiers.
8. Grad-CAM generation for the strongest model.
9. Summary table and report plot generation.

Default settings reproduce the current experiment scale:

- `IMAGE_SIZE=128`
- `CLASSIFIER_EPOCHS=5`
- `FINETUNE_EPOCHS=8`
- `FINETUNE_LR=3e-5`
- `SUPCON_EPOCHS=5`
- `NUM_WORKERS=0`

The settings can be changed without editing the script:

```bash
CLASSIFIER_EPOCHS=10 SUPCON_EPOCHS=20 ./run_full_pipeline.sh
```

To use a specific Python executable:

```bash
PYTHON_BIN=/path/to/python ./run_full_pipeline.sh
```

## Step-by-Step Experiment Commands

The following commands rerun individual parts of the study.

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

Train the ImageNet-pretrained frozen ResNet18 model:

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

Fine-tune the full ResNet18 model from the frozen-transfer checkpoint:

```bash
python src/train_classifier.py \
  --model-name resnet18 \
  --init-checkpoint result/classifier/resnet18_pretrained_frozen_128_e5/best_model.pt \
  --image-size 128 \
  --batch-size 32 \
  --epochs 8 \
  --lr 3e-5 \
  --num-workers 0 \
  --patience 4 \
  --experiment-name resnet18_unfrozen_from_frozen_lr3e-5_128_e8
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
  --checkpoint result/classifier/resnet18_unfrozen_from_frozen_lr3e-5_128_e8/best_model.pt \
  --split test \
  --num-images 12
```

Summarize results and generate report plots:

```bash
python src/summarize_results.py
python result/make_summary_plots.py
```

## Version Control Notes

The repository is configured so that large local artifacts are not committed:

- `data/`
- model checkpoints
- detailed experiment folders such as `result/classifier/`, `result/evaluation/`, `result/gradcam/`, and `result/supcon/`

The repository keeps only the source code, reproducibility scripts, summary tables, and summary plots needed to understand and regenerate the research workflow.
