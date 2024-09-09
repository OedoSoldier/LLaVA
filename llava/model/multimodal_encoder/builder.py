import os
import torch
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, AlphaCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    use_alpha = getattr(vision_tower_cfg, "alpha", False)
    use_dual = getattr(vision_tower_cfg, "dual", False)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        if use_dual:
            return torch.nn.ModuleList(
                [
                    CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs),
                    AlphaCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs),
                ]
            )
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        if use_alpha:
            return AlphaCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
