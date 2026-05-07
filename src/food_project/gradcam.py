"""Grad-CAM implementation for the classifier models."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .paths import RESULT_DIR, ensure_dir

os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(RESULT_DIR / ".matplotlib")))


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or find_last_conv_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class, probs[0].detach().cpu().numpy()


def overlay_cam_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    import matplotlib.cm as cm

    image = image.convert("RGB")
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    heatmap = cm.get_cmap("jet")(np.asarray(cam_image) / 255.0)[..., :3]
    heatmap = Image.fromarray(np.uint8(heatmap * 255))
    return Image.blend(image, heatmap, alpha=alpha)


def save_gradcam_overlay(
    model: nn.Module,
    image_path: str | Path,
    transform,
    output_path: str | Path,
    device: torch.device,
    image_size: int,
    target_class: int | None = None,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    gradcam = GradCAM(model)
    try:
        cam, resolved_target, probs = gradcam(image_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    display_image = image.resize((image_size, image_size))
    overlay = overlay_cam_on_image(display_image, cam)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "target_class": int(resolved_target),
        "probabilities": probs.tolist(),
    }
