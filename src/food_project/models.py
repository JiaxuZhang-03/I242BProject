"""Model builders for CNN baselines, transfer learning, and SupCon."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .paths import RESULT_DIR, ensure_dir

os.environ.setdefault("TORCH_HOME", str(ensure_dir(RESULT_DIR / ".torch")))

from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


MODEL_NAMES = (
    "simple_cnn",
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
)


class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 256
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ClassifierModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class SupConModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)


def _build_torchvision_encoder(
    model_name: str,
    pretrained: bool = False,
) -> tuple[nn.Module, int]:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unsupported torchvision model: {model_name}")


def build_encoder(
    model_name: str = "simple_cnn",
    pretrained: bool = False,
) -> tuple[nn.Module, int]:
    if model_name == "simple_cnn":
        encoder = SimpleCNNEncoder()
        return encoder, encoder.feature_dim
    if model_name in MODEL_NAMES:
        return _build_torchvision_encoder(model_name, pretrained=pretrained)
    raise ValueError(f"Unknown model_name={model_name}. Choices: {MODEL_NAMES}")


def build_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> ClassifierModel:
    encoder, feature_dim = build_encoder(model_name, pretrained=pretrained)
    if freeze_backbone:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    return ClassifierModel(encoder, feature_dim, num_classes, dropout=dropout)


def build_supcon_model(
    model_name: str,
    pretrained: bool = False,
    projection_dim: int = 128,
) -> SupConModel:
    encoder, feature_dim = build_encoder(model_name, pretrained=pretrained)
    return SupConModel(encoder, feature_dim, projection_dim=projection_dim)


def load_supcon_encoder(
    classifier: ClassifierModel,
    checkpoint_path: str | Path,
    strict: bool = False,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("encoder_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    return classifier.encoder.load_state_dict(state_dict, strict=strict)
