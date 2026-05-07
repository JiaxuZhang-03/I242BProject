# INDENG 1/242B Spring 2026 Project

This repository is for INDENG 1/242B Spring 2026 final project.

## Project layout

- `data/`: local image data, ignored by Git except placeholder files.
- `result/`: generated EDA figures, metrics, checkpoints, and Grad-CAM outputs.
- `src/food_project/`: shared dataset, model, training, metric, plotting, and Grad-CAM logic.
- `src/run_eda.py`: dataset summaries and sample visualizations.
- `src/train_classifier.py`: simple CNN, pretrained backbone, or SupCon-initialized classifier training.
- `src/pretrain_supcon.py`: supervised contrastive pretraining.
- `src/evaluate_model.py`: checkpoint evaluation on train/val/test splits.
- `src/make_gradcam.py`: Grad-CAM visual explanations.

## Quick commands

```bash
python src/run_eda.py
python src/train_classifier.py --model-name simple_cnn --epochs 10
python src/train_classifier.py --model-name resnet18 --pretrained --epochs 10
python src/pretrain_supcon.py --model-name resnet18 --epochs 20
python src/train_classifier.py --model-name resnet18 --supcon-checkpoint result/supcon/<run>/best_supcon.pt
python src/evaluate_model.py --checkpoint result/classifier/<run>/best_model.pt
python src/make_gradcam.py --checkpoint result/classifier/<run>/best_model.pt --num-images 12
python src/summarize_results.py
python result/make_summary_plots.py
```
