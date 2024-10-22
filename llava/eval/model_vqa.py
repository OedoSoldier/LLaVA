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
    DEFAULT_OBJ_START_TOKEN,
    DEFAULT_OBJ_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)

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


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def apply_exif_orientation(image):
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
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        image_path = os.path.join(args.image_folder, image_file)
        seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
        id_path = seg_file.replace(".npz", "_id.json")
        ids = json.load(open(id_path, "r"))
        ids = sorted(ids)

        image = Image.open(image_path)
        image = apply_exif_orientation(image)
        image = image.convert("RGBA")
        # if getattr(model.config, "image_aspect_ratio", None) == "pad":
        #     image = expand2square(
        #         image, tuple(int(x * 255) for x in image_processor.image_mean)
        #     )
        image_size = image.size
        seg = np.load(seg_file)["seg"]
        # if len(ids) == 0:
        #     ids = [0]
        #     seg = np.zeros_like(seg)

        segs = []
        bboxes = []
        segs.append(image.copy())
        h, w = image.height, image.width
        for i in ids:
            cur_seg = seg == i
            mask = Image.fromarray(np.uint8(cur_seg * 255), "L")
            # if getattr(model.config, "image_aspect_ratio", None) == "pad":
            #     mask = expand2square(mask, 0)
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

        image = segs

        image_tensor = process_images(image, image_processor, model.config)
        image_tensor = [
            x.to(dtype=torch.float16).to(device="cuda", non_blocking=True)
            for x in image_tensor
        ]
        bboxes = [
            x.to(dtype=torch.float16).to(device="cuda", non_blocking=True)
            for x in bboxes
        ]

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + DEFAULT_OBJ_START_TOKEN
                + DEFAULT_IMAGE_TOKEN * len(ids)
                + DEFAULT_OBJ_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN * len(ids) + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device="cuda", non_blocking=True)
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=([image_tensor], [bboxes]),
                image_sizes=image_size,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
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
        ans_file.flush()
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
    args = parser.parse_args()

    eval_model(args)
