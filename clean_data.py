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
DATA_FOLDER = os.path.dirname(args.data_path)


def process_file(data):
    if "image" in data:
        if type(data["image"]) is list:
            assert len(data["image"]) == 1, print(data)
            data["image"] = data["image"][0]
        image_file = data["image"]
        image_path = os.path.join(FOLDER, image_file)
        seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
        seg_info = seg_file.replace(".npz", "_id.json")
        with open(seg_info, "r") as f:
            info = json.load(f)
        # sort info by score
        try:
            # info = sorted(info, key=lambda x: x[0], reverse=False)
            seg = np.load(seg_file)["seg"]
            # w, h = seg.shape
            new_ids = []
            pixels = []
            for id, score in info:
                mask = seg == id
                total_pixels = np.sum(mask)
                new_ids.append([id, score])
                pixels.append(total_pixels)
            # sort by area
            info = [
                x
                for _, x in sorted(
                    zip(pixels, new_ids), key=lambda x: x[0], reverse=True
                )
            ]
            # print(pixels, new_ids, info)
        except:
            print(info)
            # remove seg info file
            # os.remove(seg_info)
            return None

        data["info"] = info
        # data["image"] = image_path
        # data["seg"] = seg_file

        user_inputs = data["conversations"][0]["value"]
        user_inputs = user_inputs.replace(
            DEFAULT_IMAGE_TOKEN,
            "".join([DEFAULT_IMAGE_TOKEN] * len(data["info"])),
        )
        data["conversations"][0]["value"] = user_inputs
    return data


def main():
    # data = json.load(open(f"{FOLDER}/blip_laion_cc_sbu_558k.json", "r"))
    with open(args.data_path, "r") as f:
        data = json.load(f)
    print(len(data))
    with Pool(args.num_workers) as p:
        result = list(tqdm(p.imap(process_file, data), total=len(data)))

    result = [i for i in result if i is not None]
    print(len(result))
    json.dump(
        result,
        open(f"{DATA_FOLDER}/{DATA_PATH}_cleaned.json", "w"),
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
