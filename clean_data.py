import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=cpu_count())
args = parser.parse_args()

DEFAULT_IMAGE_TOKEN = "<image>"

FOLDER = args.image_folder
DATA_PATH = os.path.basename(args.data_path).split(".")[0]


def process_file(data):
    if "image" in data:
        image_file = data["image"]
        image_path = os.path.join(FOLDER, image_file)
        seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
        seg_info = seg_file.replace(".npz", ".json")
        with open(seg_info, "r") as f:
            info = json.load(f)
        ids = [0] + [i["id"] for i in info]
        seg = np.load(seg_file)["seg"]
        w, h = seg.shape
        new_ids = []
        for id in ids:
            mask = seg == id
            total_pixels = np.sum(mask) / (w * h)
            if id == 0 and total_pixels <= 0.1:
                continue
            new_ids.append(id)
        # ids = sorted(pixels, key=lambda x: pixels[x], reverse=True)

        data["ids"] = new_ids
        # data["image"] = image_path
        # data["seg"] = seg_file

        user_inputs = data["conversations"][0]["value"]
        user_inputs = user_inputs.replace(
            DEFAULT_IMAGE_TOKEN,
            "".join([DEFAULT_IMAGE_TOKEN] * len(data["ids"])),
        )
        data["conversations"][0]["value"] = user_inputs
    return data


def main():
    # data = json.load(open(f"{FOLDER}/blip_laion_cc_sbu_558k.json", "r"))
    with open(f"{FOLDER}/{DATA_PATH}.json", "r") as f:
        data = json.load(f)
    print(len(data))
    with Pool(args.num_workers) as p:
        result = list(tqdm(p.imap(process_file, data), total=len(data)))

    result = [i for i in result if i is not None]
    print(len(result))
    json.dump(
        result,
        open(f"{FOLDER}/{DATA_PATH}_cleaned.json", "w"),
        indent=4,
    )


if __name__ == "__main__":
    main()


# 异步多线程
# import asyncio
# import os
# from tqdm.asyncio import tqdm
# import json
# import re

# DEFAULT_IMAGE_TOKEN = "<image>"

# FOLDER = "../../data/LLaVA-Pretrain"


# async def process_file(data):
#     if "image" in data:
#         image_file = data["image"]
#         image_path = os.path.join(FOLDER, image_file)
#         seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
#         id_path = seg_file.replace(".npz", "_id.json")
#         ids = json.load(open(id_path, "r"))
#         ids = sorted(ids)

#         data["ids"] = ids
#         # data["image"] = image_path
#         # data["seg"] = seg_file

#         user_inputs = data["conversations"][0]["value"]
#         user_inputs = user_inputs.replace(
#             DEFAULT_IMAGE_TOKEN,
#             "".join([DEFAULT_IMAGE_TOKEN] * len(ids)),
#         )
#         data["conversations"][0]["value"] = user_inputs
#     return data


# async def main():
#     data = json.load(open(f"{FOLDER}/blip_laion_cc_sbu_558k.json", "r"))
#     print(len(data))
#     tasks = []
#     for i in data:
#         tasks.append(process_file(i))
#     result = await tqdm.gather(*tasks)
#     result = [i for i in result if i is not None]
#     print(len(result))
#     json.dump(
#         result,
#         open(f"{FOLDER}/blip_laion_cc_sbu_558k_cleaned.json", "w"),
#         indent=4,
#     )


# if __name__ == "__main__":
#     asyncio.run(main())
