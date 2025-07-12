import torch
import torch.nn as nn
import re
from .flamingo_pytorch import PerceiverResampler


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class ShapeProjector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        # 输入: (batch_size, 1, 224, 224) - binary mask
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)  # -> (batch_size, 32, 112, 112)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # -> (batch_size, 64, 56, 56)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 128, 28, 28)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> (batch_size, 256, 14, 14)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        # 全局平均池化 -> (batch_size, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 最终投影到目标hidden_size
        self.final_proj = nn.Linear(256, hidden_size)
        
    def forward(self, x):
        # x: (batch_size, 1, 224, 224) - binary mask
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        
        # 全局平均池化
        x = self.global_avg_pool(x)  # -> (batch_size, 256, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # -> (batch_size, 256)
        
        # 投影到目标维度
        x = self.final_proj(x)  # -> (batch_size, hidden_size)
        
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    use_dual = getattr(config, "dual", False)

    if projector_type == "linear":
        if use_dual:
            return torch.nn.ModuleList(
                [
                    nn.Linear(config.mm_hidden_size, config.hidden_size),
                    nn.Linear(config.mm_hidden_size, config.hidden_size),
                ]
            )
        else:
            return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        if use_dual:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [
                [nn.Linear(config.mm_hidden_size, config.hidden_size)],
                [nn.Linear(config.mm_hidden_size, config.hidden_size)],
            ]
            for _ in range(1, mlp_depth):
                modules[0].append(nn.GELU())
                modules[0].append(nn.Linear(config.hidden_size, config.hidden_size))
                modules[1].append(nn.GELU())
                modules[1].append(nn.Linear(config.hidden_size, config.hidden_size))
            return torch.nn.ModuleList(
                [nn.Sequential(*modules[0]), nn.Sequential(*modules[1])]
            )
        else:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [
                    PerceiverResampler(
                        dim=config.mm_hidden_size,
                        depth=mlp_depth,
                        dim_head=64,
                        heads=8,
                        num_latents=32,
                    ),
                    nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


def build_shape_projector(config):
    # a cnn for capturing shape features from binary mask
    shape_projector = ShapeProjector(config.hidden_size)
    return shape_projector

def build_confidence_projector(config):
    confidence_projector = [
        nn.Linear(1, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.GELU(),
        nn.Linear(config.hidden_size, config.hidden_size),
    ]
    return nn.Sequential(*confidence_projector)