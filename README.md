# INDENG 1/242B Spring 2026 Project

## Research Overview

This repository implements a reproducible image classification study for the INDENG 1/242B Spring 2026 final project. The task is binary food image classification: given an image, the model predicts whether the food is `healthy` or `unhealthy`.

The study is organized as an incremental experimental pipeline rather than a single model training script. It begins with exploratory data analysis, establishes a simple convolutional neural network baseline, evaluates transfer learning with an ImageNet-pretrained ResNet18 backbone, improves the transfer model through full fine-tuning, and includes a supervised contrastive learning branch for representation learning. After the main accuracy improvement step, the project adds two research extensions: error analysis of the final model and a focused ResNet18 fine-tuning ablation study. The final pipeline also generates evaluation tables, training curves, confusion matrices, ROC curves, and Grad-CAM visual explanations for model interpretability.

The strongest model in the current study is a fully fine-tuned ResNet18 initialized from a frozen-transfer checkpoint and trained with learning rate `1e-4`. It achieves `0.9617` test accuracy and `0.9944` ROC AUC on the held-out test set.

## Research Questions

This project investigates three practical questions in food image classification:

1. How much performance can be obtained from a small task-specific CNN trained from scratch?
2. How much does ImageNet transfer learning improve classification performance under the same train/validation/test protocol?
3. Does further representation adaptation, through either full backbone fine-tuning or supervised contrastive pretraining, improve held-out test performance and provide stronger report-ready evidence?
4. Which part of the transfer-learning strategy explains the improvement: unfreezing only the high-level ResNet block, changing the full fine-tuning learning rate, or updating the entire backbone?
5. What kinds of samples remain difficult after the best model is selected, and are residual errors balanced across the two classes?

The experimental design treats the simple CNN as a baseline, the frozen ResNet18 as the primary transfer learning comparison, the fully fine-tuned ResNet18 as the main accuracy-improvement experiment, and the SupCon branch as an additional representation-learning comparison. The ablation and error-analysis scripts then examine why the best ResNet18 setting works and where it still fails.

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
   The best frozen ResNet18 checkpoint is used to initialize a second training stage. The backbone is then unfrozen and optimized with small learning rates (`1e-5`, `3e-5`, and `1e-4` in the ablation study). This tests whether adapting pretrained features to the food dataset improves generalization.

4. **Supervised contrastive pretraining and fine-tuning**
   A supervised contrastive loss is used to pretrain an encoder with label-aware representation learning. The encoder is then reused for classifier fine-tuning. This provides an alternative representation-learning path for comparison with ImageNet transfer learning.

5. **Error analysis and ablation**
   After identifying the strongest model, the project exports the remaining misclassified samples, confidence scores, class-specific error rates, and the most confident errors. A separate ablation summary compares the frozen classifier head, ResNet layer4 fine-tuning, and full-backbone fine-tuning under different learning rates.

For evaluation, each trained classifier is assessed on the held-out test split. The summary pipeline reports accuracy, macro F1, weighted F1, ROC AUC, confusion matrices, and training history plots. Grad-CAM is generated for the strongest model to support qualitative interpretability.

## Current Results

The current test-set results are:

| Experiment | Best validation accuracy | Test accuracy | Macro F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 full fine-tune, LR 1e-4 | 0.9655 | 0.9617 | 0.9617 | 0.9944 |
| ResNet18 layer4 fine-tune, LR 3e-5 | 0.9540 | 0.9406 | 0.9406 | 0.9850 |
| ResNet18 full fine-tune, LR 3e-5 | 0.9598 | 0.9406 | 0.9406 | 0.9845 |
| ResNet18 full fine-tune, LR 1e-5 | 0.9406 | 0.9310 | 0.9310 | 0.9781 |
| ResNet18 pretrained frozen | 0.8563 | 0.8697 | 0.8697 | 0.9344 |
| Simple CNN + SupCon fine-tune | 0.8084 | 0.8046 | 0.8033 | 0.8813 |
| Simple CNN baseline | 0.7989 | 0.8027 | 0.8027 | 0.8869 |

The strongest improvement comes from unfreezing and fine-tuning ResNet18 after the frozen-transfer stage. Compared with the frozen ResNet18 model, full fine-tuning with learning rate `1e-4` improves test accuracy from `0.8697` to `0.9617`, an absolute increase of approximately `9.20` percentage points. The final model also improves ROC AUC from `0.9344` to `0.9944`.

The final ResNet18 fine-tuned model has the following held-out test confusion matrix:

| True class | Predicted healthy | Predicted unhealthy |
| --- | ---: | ---: |
| Healthy | 251 | 10 |
| Unhealthy | 10 | 251 |

This indicates balanced performance across the two classes: healthy recall is `0.9617`, unhealthy recall is `0.9617`, and both class-level F1 scores are approximately `0.962`.

## Error Analysis and Ablation Findings

The error analysis of the final model shows `20` misclassified samples out of `522` test images. Errors are evenly distributed across classes: `10` healthy images are predicted as unhealthy, and `10` unhealthy images are predicted as healthy. The mean confidence among errors is `0.7843`, and `12` of the `20` errors have predicted-class probability at least `0.80`, suggesting that the most informative remaining failures are not merely low-confidence borderline cases.

The ResNet18 ablation study supports three conclusions:

1. Freezing the ImageNet backbone and training only the classifier head is useful but limited, reaching `0.8697` test accuracy.
2. Unfreezing only ResNet `layer4` reaches `0.9406` test accuracy, nearly matching the earlier full fine-tuning result at learning rate `3e-5`. This suggests that much of the improvement comes from adapting high-level semantic features.
3. Full-backbone fine-tuning is sensitive to learning rate. In this setup, `1e-5` under-adapts (`0.9310` test accuracy), `3e-5` reaches `0.9406`, and `1e-4` gives the strongest result (`0.9617`).

## Generated Research Artifacts

The pipeline creates the following report-ready artifacts:

- EDA outputs: class counts, split summaries, and sample image grids.
- Training outputs: checkpoints, history CSV files, and per-experiment training curves.
- Evaluation outputs: metrics JSON files, predictions CSV files, classification reports, confusion matrices, and ROC curves.
- Interpretability outputs: Grad-CAM visualizations for the strongest model.
- Error-analysis outputs: misclassified sample tables, confidence distributions, class error rates, and most confident error grids.
- Ablation outputs: focused fine-tuning comparison tables and learning-rate sensitivity plots.
- Summary outputs: experiment summary tables and final plots under `result/summary_plots/`.

The lightweight summary artifacts are kept in Git:

- `result/experiment_summary.csv`
- `result/experiment_summary.md`
- `result/ablation_summary.csv`
- `result/ablation_summary.md`
- `result/error_analysis_summary.csv`
- `result/error_analysis_summary.md`
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
- `src/error_analysis.py`: analyzes misclassified samples and confidence patterns from evaluation predictions.
- `src/summarize_ablation.py`: creates focused ResNet18 fine-tuning ablation tables and plots.
- `download_data.sh`: downloads and validates the dataset into `data/`.
- `setup_environment.sh`: one-command environment setup using `requirements.txt`.
- `run_full_pipeline.sh`: one-command script that runs the full research pipeline.
- `run_resnet_ablation.sh`: one-command script for the focused ResNet18 fine-tuning ablation.
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
- `FINETUNE_LR=1e-4`
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
  --lr 1e-4 \
  --num-workers 0 \
  --patience 4 \
  --experiment-name resnet18_unfrozen_from_frozen_lr1e-4_128_e8
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
  --checkpoint result/classifier/resnet18_unfrozen_from_frozen_lr1e-4_128_e8/best_model.pt \
  --split test \
  --num-images 12
```

Summarize results and generate report plots:

```bash
python src/summarize_results.py
python result/make_summary_plots.py
```

Run error analysis for the strongest model:

```bash
python src/error_analysis.py \
  --experiment resnet18_unfrozen_from_frozen_lr1e-4_128_e8
```

Run the ResNet18 ablation study:

```bash
./run_resnet_ablation.sh
```

If the ablation experiments already exist and only the table/plots need to be rebuilt:

```bash
python src/summarize_ablation.py
```

## Version Control Notes

The repository is configured so that large local artifacts are not committed:

- `data/`
- model checkpoints
- detailed experiment folders such as `result/classifier/`, `result/evaluation/`, `result/gradcam/`, and `result/supcon/`

The repository keeps only the source code, reproducibility scripts, summary tables, and summary plots needed to understand and regenerate the research workflow.
