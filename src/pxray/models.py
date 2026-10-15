"""Models: a small CNN from scratch and an ImageNet MobileNetV3-Small adapted to grayscale."""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SmallCNN(nn.Module):
    """Four conv blocks (16-32-64-96 channels), global average pooling, one logit."""

    def __init__(self, in_ch: int = 1, widths: tuple[int, ...] = (16, 32, 64, 96)) -> None:
        super().__init__()
        chans = (in_ch, *widths)
        self.features = nn.Sequential(
            *[_block(chans[i], chans[i + 1], pool=i < len(widths) - 1) for i in range(len(widths))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(f)).squeeze(1)


class MobileNetGray(nn.Module):
    """torchvision MobileNetV3-Small, grayscale stem and a single-logit head.

    The pretrained RGB stem is collapsed by summing its filters over the colour axis, which is
    equivalent to feeding a gray image replicated on three channels. The last stride-2 stage is
    set to stride 1 so the Grad-CAM map is 1/16 of the input size (8x8 at 128 px, not 4x4).
    """

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        old = net.features[0][0]
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        net.features[0][0] = new
        for m in net.features[9].modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


MODEL_NAMES = ("small_cnn", "mobilenet_v3_small")


def build_model(name: str, pretrained: bool = False) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    if name == "mobilenet_v3_small":
        return MobileNetGray(pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional feature block (spatial map before pooling)."""
    if isinstance(model, SmallCNN):
        return model.features[-1]
    if isinstance(model, MobileNetGray):
        return model.net.features[-1]
    raise ValueError("unsupported model")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["MODEL_NAMES", "MobileNetGray", "SmallCNN", "build_model", "count_parameters",
           "gradcam_layer", "torch"]
