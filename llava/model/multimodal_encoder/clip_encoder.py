import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

import os
import types
import collections
import wget


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                "Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git"
            )
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size[
                "width"
            ] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size[
            "width"
        ] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                )
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)


def rewrited_forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    mask_torch = pixel_values[:, [3], :, :]
    pixel_values = pixel_values[:, :3, :, :]

    batch_size = pixel_values.shape[0]
    patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]

    alpha = mask_torch * 1.9231
    patch_embeds = patch_embeds + self.patch_embedding_alpha(alpha)
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


class AlphaCLIPVisionTower(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)
        self.select_feature = "cls"

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )

        # -----------------------------------------
        visual_encoder = self.vision_tower.vision_model

        visual_encoder.embeddings.patch_embedding_alpha = torch.nn.Conv2d(
            in_channels=1,
            out_channels=visual_encoder.embeddings.patch_embedding.out_channels,
            kernel_size=visual_encoder.embeddings.patch_embedding.kernel_size,
            stride=visual_encoder.embeddings.patch_embedding.stride,
            bias=False,
        ).to(visual_encoder.embeddings.patch_embedding.weight.device)
        visual_encoder.embeddings.forward = types.MethodType(
            rewrited_forward, visual_encoder.embeddings
        )

        filename = "clip_l14@336_grit1m_fultune_8xe.pth"
        if not os.path.exists(filename):
            filename = wget.download(
                "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight//clip_l14_336_grit1m_fultune_8xe.pth"
            )
        print(f"Loading pretrained Alpha-CLIP model from {filename}")

        state_dict = torch.load("clip_l14@336_grit1m_fultune_8xe.pth")
        converted_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if "transformer.resblocks" in k:
                new_key = (
                    k.replace("transformer.resblocks", "encoder.layers")
                    .replace("attn", "self_attn")
                    .replace("ln_1", "layer_norm1")
                    .replace("ln_2", "layer_norm2")
                    .replace("c_fc", "fc1")
                    .replace("c_proj", "fc2")
                )
                if ("self_attn" in new_key) and (
                    "out" not in new_key
                ):  # split qkv attn
                    if "weight" in new_key:
                        converted_dict[new_key.replace("in_proj", "q_proj")] = v[
                            :1024, :
                        ]
                        converted_dict[new_key.replace("in_proj", "k_proj")] = v[
                            1024:2048, :
                        ]
                        converted_dict[new_key.replace("in_proj", "v_proj")] = v[
                            2048:, :
                        ]
                    else:
                        assert "bias" in new_key
                        converted_dict[new_key.replace("in_proj", "q_proj")] = v[:1024]
                        converted_dict[new_key.replace("in_proj", "k_proj")] = v[
                            1024:2048
                        ]
                        converted_dict[new_key.replace("in_proj", "v_proj")] = v[2048:]
                else:
                    converted_dict[new_key] = v
            else:
                new_key = (
                    k.replace("class_embedding", "embeddings.class_embedding")
                    .replace("conv1.weight", "embeddings.patch_embedding.weight")
                    .replace(
                        "positional_embedding", "embeddings.position_embedding.weight"
                    )
                    .replace(
                        "conv1_alpha.weight", "embeddings.patch_embedding_alpha.weight"
                    )
                    .replace("ln_pre.weight", "pre_layrnorm.weight")
                    .replace("ln_pre.bias", "pre_layrnorm.bias")
                    .replace("ln_post.weight", "post_layernorm.weight")
                    .replace("ln_post.bias", "post_layernorm.bias")
                )
                converted_dict[new_key] = v

        visual_encoder.load_state_dict(converted_dict, strict=False)
        self.vision_tower.vision_model = visual_encoder

        if len(self.image_processor.image_mean) < 4:
            self.image_processor.image_mean.append(0.5)
            self.image_processor.image_std.append(0.5)
        self.image_processor.do_convert_rgb = False
        # -----------------------------------------
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "cls":
            # image_features = image_features[:, 0].unsqueeze(1)
            image_features = image_features[:, 1:].mean(dim=1).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features
