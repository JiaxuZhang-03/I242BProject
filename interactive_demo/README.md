# Interactive Demo

This folder contains a lightweight food-image health classifier demo. It serves a polished upload page and runs inference with the best ResNet18 checkpoint from the project.

## Run

From the repository root:

```bash
./interactive_demo/run_demo.sh
```

Then open:

```text
http://127.0.0.1:7860
```

The demo uses:

```text
result/classifier/resnet18_unfrozen_from_frozen_lr1e-4_128_e8/best_model.pt
```

To use another trained classifier:

```bash
CHECKPOINT_PATH=/path/to/best_model.pt ./interactive_demo/run_demo.sh
```

Useful options:

```bash
./interactive_demo/run_demo.sh --port 8000 --device cpu
```

The launcher uses `/Users/littleotter/miniconda3/envs/nnenv2/bin/python` by default. No extra web framework is required; the server uses the Python standard library plus the existing PyTorch/Pillow/torchvision project dependencies.
