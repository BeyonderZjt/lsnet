from typing import Dict

import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis
from model.lsnet import LSNet
from model.ska import SKA


class DepthwiseSeparableConv3x3(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return self.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        cat = torch.cat([avg_map, max_map], dim=1)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels=channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


LIGHTWEIGHT_CFGS: Dict[str, Dict] = {
    "lsnet_ef_nano": {
        "embed_dim": [48, 96, 160, 224],
        "depth": [1, 1, 3, 3],
        "num_heads": [2, 2, 2, 4],
        "key_dim": [12, 12, 12, 12],
    },
    "lsnet_ef_micro": {
        "embed_dim": [56, 112, 192, 256],
        "depth": [1, 2, 4, 4],
        "num_heads": [2, 2, 3, 4],
        "key_dim": [12, 12, 16, 16],
    },
    "lsnet_ef_tiny": {
        "embed_dim": [32, 64, 96, 128],
        "depth": [1, 1, 2, 2],
        "num_heads": [2, 2, 2, 2],
        "key_dim": [8, 8, 8, 8],
    },
}


class LSNetMultiDomainEF(nn.Module):
    """
    多域输入前融合版本：
    1) Doppler/Range 分别经过轻量化 3x3 深度可分离卷积（保持 3 通道和分辨率）
    2) 在通道维拼接为 6 通道
    3) 输入改造后的轻量化 LSNet 主干
    4) 进入分类层前加入 CBAM
    """

    def __init__(
        self,
        num_classes: int = 6,
        variant: str = "lsnet_ef_nano",
        img_size: int = 224,
        patch_size: int = 8,
        pretrained: bool = False,
    ):
        super().__init__()
        if variant not in LIGHTWEIGHT_CFGS:
            raise ValueError(f"Unknown variant={variant}. Available: {list(LIGHTWEIGHT_CFGS.keys())}")

        cfg = LIGHTWEIGHT_CFGS[variant]
        self.pre_doppler = DepthwiseSeparableConv3x3(channels=3)
        self.pre_range = DepthwiseSeparableConv3x3(channels=3)

        # 使用 LSNet 原主干代码，改 in_chans=6 且取消原分类头
        self.backbone = LSNet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=6,
            num_classes=0,
            embed_dim=cfg["embed_dim"],
            key_dim=cfg["key_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            distillation=False,
        )

        out_channels = cfg["embed_dim"][-1]
        self.cbam = CBAM(channels=out_channels, reduction=16)
        self.classifier = nn.Linear(out_channels, num_classes)

        if pretrained:
            print("[Warning] pretrained=True is ignored in this custom fused architecture.")

    def forward(self, doppler: torch.Tensor, range_tensor: torch.Tensor) -> torch.Tensor:
        doppler = self.pre_doppler(doppler)
        range_tensor = self.pre_range(range_tensor)
        x = torch.cat([doppler, range_tensor], dim=1)  # [B, 6, H, W]

        x = self.backbone.patch_embed(x)
        x = self.backbone.blocks1(x)
        x = self.backbone.blocks2(x)
        x = self.backbone.blocks3(x)
        x = self.backbone.blocks4(x)

        x = self.cbam(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x


def _ska_forward_for_flops(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Reuse the module's PyTorch fallback implementation for stable FLOPs tracing.
    return SKA._fallback(x, w)


def count_flops(model: nn.Module, image_size: int = 224) -> int:
    """Return FLOPs count (for one forward) using fvcore FlopCountAnalysis."""
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    was_training = model.training
    model_cpu = model.to("cpu").eval()
    input_d = torch.randn(1, 3, image_size, image_size)
    input_r = torch.randn(1, 3, image_size, image_size)

    old_ska_forward = SKA.forward
    SKA.forward = _ska_forward_for_flops
    try:
        flops = (
            FlopCountAnalysis(model_cpu, (input_d, input_r))
            .unsupported_ops_warnings(False)
            .uncalled_modules_warnings(False)
            .total()
        )
    finally:
        SKA.forward = old_ska_forward
        model.to(original_device)
        model.train(was_training)

    return int(flops)
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def build_lsnet_multidomain_ef(num_classes: int = 6, variant: str = "lsnet_ef_nano") -> LSNetMultiDomainEF:
    return LSNetMultiDomainEF(num_classes=num_classes, variant=variant)


if __name__ == "__main__":
    model = build_lsnet_multidomain_ef(num_classes=6, variant="lsnet_ef_nano")
    stats = count_parameters(model)
    flops = count_flops(model, image_size=224)
    print(
        f"Params(total/trainable/non-trainable): "
        f"{stats['total']:,} / {stats['trainable']:,} / {stats['non_trainable']:,}"
    )
    print(f"FLOPs (1x224x224 dual-branch input): {flops:,}")
    d = torch.randn(2, 3, 224, 224)
    r = torch.randn(2, 3, 224, 224)
    y = model(d, r)
    print("Output shape:", y.shape)
