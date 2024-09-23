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
from torch.cuda.amp import autocast

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
from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# set warning level
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加start,end 的 arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--data-path", type=str, default=None)
parser.add_argument("--image-folder", type=str, default=None)
args = parser.parse_args()

meta = args.data_path
# data = [json.loads(q) for q in open(meta, "r")]
# data = json.load(open(meta, "r"))
if meta.endswith(".jsonl"):
    data = [json.loads(q) for q in open(meta, "r")]
else:
    data = json.load(open(meta, "r"))
image_folder = args.image_folder
input_paths = []
for i in data:
    if "image" in i.keys():
        image_path = os.path.join(image_folder, i["image"])
        seg_path = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
        id_path = seg_path.replace(".npz", "_id.json")
        if not (os.path.exists(id_path) and os.path.exists(seg_path)):
            if os.path.exists(image_path):
                input_paths.append(image_path)
            else:
                print(f"{image_path} not exists")
# 去重
input_paths = list(set(input_paths))
print(len(input_paths))
if len(input_paths) == 0:
    print("All images are processed")
    exit()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


input_paths = get_chunk(input_paths, args.num_chunks, args.chunk_idx)
print(len(input_paths))
# exit()
# warnings.filterwarnings("ignore")

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


class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        original_image = original_image.cpu().numpy()  # (B, H, W, C)
        inputs = []
        for i in range(original_image.shape[0]):
            height, width = original_image[i].shape[:2]
            aug_input = T.AugInput(original_image[i], sem_seg=None)
            self.aug(aug_input)
            image = aug_input.image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            temp_input = {"image": image, "height": height, "width": width}
            inputs.append(temp_input)
        # logger.info("forwarding")
        with autocast():
            predictions = self.model(inputs)
        # logger.info("done")
        return predictions

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predict(image)
        # visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        # if "panoptic_seg" in predictions:
        #     panoptic_seg, segments_info = predictions["panoptic_seg"]
        #     vis_output = visualizer.draw_panoptic_seg(
        #         panoptic_seg.to(self.cpu_device), segments_info
        #     )
        # else:
        #     if "sem_seg" in predictions:
        #         vis_output = visualizer.draw_sem_seg(
        #             predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
        #         )
        #     if "instances" in predictions:
        #         instances = predictions["instances"].to(self.cpu_device)
        #         vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


models = {}
for model_name, cfg_name in zip(
    ["ODISE(Label)"],
    ["Panoptic/odise_label_coco_50e.py"],
):

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
    # model = create_ddp_model(model)
    # ODISECheckpointer(model).load(cfg.train.init_checkpoint)
    ODISECheckpointer(model).load("./odise_label_coco_50e-b67d2efc.pth")
    models[model_name] = model


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


# def inference(image_paths, output_path, vocab, label_list, model_name):

#     logger.info("building class names")
#     demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
#     with open(os.path.join("classes.json"), "w") as f:
#         json.dump(demo_classes, f, indent=4)
#     with open(os.path.join("metadata.json"), "w") as f:
#         json.dump(demo_metadata.as_dict(), f, indent=4)
#     if model_name is None:
#         model_name = "ODISE(Label)"
#     with ExitStack() as stack:
#         logger.info(f"loading model {model_name}")
#         inference_model = OpenPanopticInference(
#             model=models[model_name],
#             labels=demo_classes,
#             metadata=demo_metadata,
#             semantic_on=False,
#             instance_on=False,
#             panoptic_on=True,
#         )
#         stack.enter_context(inference_context(inference_model))
#         stack.enter_context(torch.no_grad())

#         demo = VisualizationDemo(inference_model, demo_metadata, aug)
#         for image_path in tqdm(image_paths):
#             filename = os.path.basename(image_path)
#             if os.path.exists(
#                 os.path.join(output_path, filename.replace("jpeg", "json"))
#             ):
#                 continue
#             img = utils.read_image(image_path, format="RGB")
#             predictions, _ = demo.run_on_image(img)
#             seg, info = predictions["panoptic_seg"]
#             with open(
#                 os.path.join(output_path, filename.replace("jpeg", "json")), "w"
#             ) as f:
#                 json.dump(info, f, indent=4)
#             seg = seg.cpu().numpy()
#             # print(seg.shape)
#             np.savez_compressed(
#                 os.path.join(output_path, filename.replace("jpeg", "npz")), seg=seg
#             )
#             # flush gpu mem
#             torch.cuda.empty_cache()


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        if img_path.endswith(".gif"):
            img = Image.open(img_path)
            img.seek(0)
            img = utils.convert_PIL_to_numpy(img, format="RGB")
        else:
            img = utils.read_image(img_path, format="RGB")
        return img, img_path


def inference(image_paths, vocab, label_list, model_name):

    logger.info("building class names")
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)

    # Prepare DataLoader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    if model_name is None:
        model_name = "ODISE(Label)"

    inference_model = OpenPanopticInference(
        model=models[model_name],
        labels=demo_classes,
        metadata=demo_metadata,
        semantic_on=False,
        instance_on=False,
        panoptic_on=True,
    )
    inference_model = inference_model.cuda()

    demo = VisualizationDemo(inference_model, demo_metadata, aug)

    inference_model.eval()
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader):
            if None in inputs:
                print(f"Error: {paths}")
                continue
            inputs = inputs.cuda()
            predictions, _ = demo.run_on_image(inputs)
            for idx, prediction in enumerate(predictions):
                try:
                    seg, info = prediction["panoptic_seg"]
                    filename = paths[
                        idx
                    ]  # os.path.join(output_path, os.path.basename(paths[idx]))
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
                except:
                    print(f"Error: {paths[idx]}")
                    continue
        # flush gpu mem
        # torch.cuda.empty_cache()


# input_paths = glob(os.path.expanduser("/home/wangsihan/workspace/coco2017/images/*.jpg"))
# # image_folder = "/home/wangsihan/workspace/coco2014/images"
# # meta = json.load(open("/home/wangsihan/workspace/coco2014/qa90_questions.jsonl", "r"))
# # input_paths = [os.path.join(image_folder, i["image"]) for i in meta]
# output_path = "/home/wangsihan/workspace/coco2017/odise_segs"  # "images/output"
# cleaned_list = []
# for i in input_paths:
#     filename = os.path.basename(i)
#     if not os.path.exists(os.path.join(output_path, filename.replace("jpg", "json"))):
#         cleaned_list.append(i)
# input_paths = cleaned_list
# print(len(input_paths))
# # glob(os.path.expanduser("images/input/*.jpeg"))

# meta = "/home/wangsihan/workspace/data/LLaVA-Finetune/llava_v1_5_mix665k.json"
# data = json.load(open(meta, "r"))
# image_folder = os.path.dirname(meta)
# input_paths = []
# for i in data:
#     if "image" in i.keys():
#         image_path = os.path.join(image_folder, i["image"])
#         seg_path = re.sub(r"\.(jpg|jpeg|png|bmp)$", ".npz", image_path)
#         if not os.path.exists(seg_path):
#             if os.path.exists(image_path):
#                 input_paths.append(image_path)
#             else:
#                 print(f"{image_path} not exists")
#         else:
#             print(f"{image_path} is already processed")
# print(len(input_paths))


inference(input_paths, None, ["COCO", "ADE", "LVIS"], None)
