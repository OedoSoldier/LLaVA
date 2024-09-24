# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import itertools
import json
from contextlib import ExitStack
import torch
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.engine import create_ddp_model, default_argument_parser, hooks, launch
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from PIL import Image, ExifTags
from torch.amp import autocast

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference

from glob import glob
import argparse
from tqdm import tqdm
import os
import warnings
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

import json
import re
import math
import pandas as pd

from io import BytesIO
import base64

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# set warning level
warnings.filterwarnings("ignore", category=FutureWarning)

setup_logger()
logger = setup_logger(name="odise")

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(
        itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES)
    )
)


def load_odise():
    model_name = "ODISE(Label)"
    cfg_name = "Panoptic/odise_label_coco_50e.py"
    cfg = model_zoo.get_config(cfg_name, trained=True)

    cfg.model.overlap_threshold = 0
    cfg.model.clip_head.alpha = 0.35
    cfg.model.clip_head.beta = 0.65
    cfg.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(torch.float16)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load("./odise_label_coco_50e-b67d2efc.pth")
    return model, aug


def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [
        random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))
    ]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if any("COCO" in label for label in label_list):
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if any("ADE" in label for label in label_list):
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if any("LVIS" in label for label in label_list):
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


def load_img(image, aug):
    # try:
    #     if img_path.endswith(".gif"):
    #         img = Image.open(img_path)
    #         img.seek(0)
    #         img = utils.convert_PIL_to_numpy(img, format="RGB")
    #     else:
    #         img = utils.read_image(img_path, format="RGB")
    # except:
    #     print(f"Error: {img_path}")
    img = Image.open(BytesIO(base64.b64decode(image)))
    img = img.convert("RGB")
    img = np.asarray(img)
    height, width = img.shape[:2]
    aug_input = T.AugInput(img, sem_seg=None)
    aug(aug_input)
    image = aug_input.image
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    return [{"image": image, "height": height, "width": width}]


def inference(questions, model, aug, save_dir):
    with torch.no_grad():
        for index, row in tqdm(questions.iterrows(), total=len(questions)):
            inputs = load_img(row["image"], aug)
            with autocast("cuda"):
                predictions = model(inputs)
            seg, info = predictions[0]["panoptic_seg"]
            filename = os.path.join(save_dir, str(row["index"]) + ".jpg")
            seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", filename)
            info_file = seg_file.replace("npz", "json")
            with open(info_file, "w") as f:
                json.dump(info, f, indent=4)
            seg = seg.cpu().numpy()
            np.savez_compressed(seg_file, seg=seg)
            id_path = info_file.replace(".json", "_id.json")
            ids = [0] + [j["id"] for j in info]
            w, h = seg.shape
            new_ids = []
            for id in ids:
                mask = seg == id
                total_pixels = np.sum(mask) / (w * h)
                if id == 0 and total_pixels <= 0.1:
                    continue
                new_ids.append(id)
            ids = new_ids
            with open(id_path, "w") as f:
                json.dump(ids, f)
        # flush gpu mem
        # torch.cuda.empty_cache()


def check_exists(flag, data):
    r = []
    for i in data:
        if flag:
            r.append(i)
        else:
            seg_path = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", i)
            id_path = seg_path.replace(".npz", "_id.json")
            if not (os.path.exists(id_path) and os.path.exists(seg_path)):
                if os.path.exists(i):
                    r.append(i)
                else:
                    print(f"{i} not exists")
    return r


def get_chunk(lst, n, k):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    chunk_start = list(range(0, len(lst), chunk_size))
    chunk_end = chunk_start[1:] + [len(lst)]
    return lst[chunk_start[k] : chunk_end[k]]


def main(args):
    # input_paths = load_data(args)
    questions = pd.read_table(os.path.expanduser(args.data_path))
    filename = os.path.basename(args.data_path).split(".")[0]
    image_folder = os.path.dirname(args.data_path)
    save_dir = os.path.join(image_folder, "images", filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input_paths = []
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        filename = os.path.join(save_dir, str(row["index"]) + ".jpg")
        seg_path = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", filename)
        id_path = seg_path.replace(".npz", "_id.json")
        if not (os.path.exists(id_path) and os.path.exists(seg_path)):
            input_paths.append(row["index"])

    if len(input_paths) == 0:
        print("All images are processed")
        exit()

    input_paths = get_chunk(input_paths, args.num_chunks, args.chunk_idx)
    print(
        f"Processing chunk {args.chunk_idx}/{args.num_chunks}, chunck size: {len(input_paths)}"
    )
    questions = questions[questions["index"].isin(input_paths)]

    model, aug = load_odise()

    logger.info("building class names")
    demo_classes, demo_metadata = build_demo_classes_and_metadata(
        None, ["COCO", "ADE", "LVIS"]
    )

    inference_model = OpenPanopticInference(
        model=model,
        labels=demo_classes,
        metadata=demo_metadata,
        semantic_on=False,
        instance_on=False,
        panoptic_on=True,
    )
    inference_model = inference_model.cuda()

    # demo = VisualizationDemo(inference_model, demo_metadata, aug)
    inference_model.eval()
    inference(questions, inference_model, aug, save_dir)


if __name__ == "__main__":
    # 添加start,end 的 arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
