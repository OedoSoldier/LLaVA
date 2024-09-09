import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import re
import numpy as np


_EXIF_ORIENT = 274


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self, questions, image_folder, tokenizer, image_processor, model_config
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self._load_seg_id()

    @staticmethod
    def _apply_exif_orientation(image):
        """
        Applies the exif orientation correctly.

        This code exists per the bug:
        https://github.com/python-pillow/Pillow/issues/3973
        with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
        various methods, especially `tobytes`

        Function based on:
        https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
        https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

        Args:
            image (PIL.Image): a PIL image

        Returns:
            (PIL.Image): the PIL image with exif orientation applied, if applicable
        """
        if not hasattr(image, "getexif"):
            return image

        try:
            exif = image.getexif()
        except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
            exif = None

        if exif is None:
            return image

        orientation = exif.get(_EXIF_ORIENT)

        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)

        if method is not None:
            return image.transpose(method)
        return image

    def _load_seg_id(self):
        for i, data in tqdm(enumerate(self.questions), total=len(self.questions)):
            if "image" in data:
                image_file = data["image"]
                image_folder = self.image_folder
                image_path = os.path.join(image_folder, image_file)
                seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
                id_path = seg_file.replace(".npz", "_id.json")
                ids = json.load(open(id_path, "r"))
                ids = sorted(ids)

                self.questions[i]["image"] = image_path
                self.questions[i]["seg"] = seg_file
                self.questions[i]["ids"] = ids

    def _load_image_and_prompt(self, idx):
        line = self.questions[idx]
        image_path = line["image"]
        seg_file = line["seg"]
        ids = line["ids"]
        qs = line["text"]

        if not os.path.exists(image_path):
            cur_ext = os.path.basename(image_path).split(".")[-1]
            for ext in ["jpg", "jpeg", "png", "bmp", "gif"]:
                new_path = image_path.replace(cur_ext, ext)
                if os.path.exists(new_path):
                    image_path = new_path
                    break
        image = Image.open(image_path)
        if image_path.endswith(".gif"):
            image.seek(0)
        # image = _apply_exif_orientation(image)
        image = CustomDataset._apply_exif_orientation(image)
        image = image.convert("RGBA")
        image_size = image.size
        seg = np.load(seg_file)["seg"]
        if len(ids) == 0:
            ids = [0]
            seg = np.zeros_like(seg)

        segs = []
        bboxes = []
        # segs.append(image.copy())
        h, w = image.height, image.width
        for i in ids:
            cur_seg = seg == i
            mask = Image.fromarray(np.uint8(cur_seg * 255), "L")
            bbox = mask.getbbox()
            if bbox is None:
                bbox = [0, 0, 1, 1, 1]
            else:
                bbox = [
                    bbox[0] / w,
                    bbox[1] / h,
                    (bbox[2] - bbox[0]) / h,
                    (bbox[3] - bbox[1]) / w,
                    np.sum(cur_seg) / (h * w),
                ]  # normalize bbox
            bboxes.append(torch.tensor(bbox))
            temp = image.copy()
            temp.putalpha(mask)
            segs.append(temp)

        qs = DEFAULT_IMAGE_TOKEN * len(ids) + "\n" + qs

        if self.model_config.mm_use_im_start_end:
            qs = qs.replace(
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
            )

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt, segs, bboxes, image_size

    def __getitem__(self, index):
        prompt, image, bboxes, image_size = self._load_image_and_prompt(index)

        image_tensor = process_images(image, self.image_processor, self.model_config)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        image_tensor = [x.to(dtype=torch.float16) for x in image_tensor]
        bboxes = [x.to(dtype=torch.float16) for x in bboxes]

        return input_ids, image_tensor, bboxes, image_size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, bboxes, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    return input_ids, image_tensors, bboxes, image_sizes


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config
    )

    for (input_ids, image_tensor, bboxes, image_sizes), line in tqdm(
        zip(data_loader, questions), total=len(questions)
    ):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)
        image_tensor = list(image_tensor)
        bboxes = list(bboxes)
        for i, batch in enumerate(image_tensor):
            for image in batch:
                image.to(device="cuda", non_blocking=True)
            for bbox in bboxes[i]:
                bbox.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=(image_tensor, bboxes),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
