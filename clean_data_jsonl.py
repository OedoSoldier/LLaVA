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
DATA_PATH = args.data_path


def process_file(data):
    if "image" in data:
        image_file = data["image"]
        image_path = os.path.join(FOLDER, image_file)
        seg_file = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
        seg_info = seg_file.replace(".npz", ".json")
        id_file = seg_file.replace(".npz", "_id.json")
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
        if len(info) == 0:
            print(image_file, seg)
        # ids = sorted(pixels, key=lambda x: pixels[x], reverse=True)
        # print(id_file)
        with open(id_file, "w") as f:
            json.dump(ids, f)
    return data


def main():
    # data = json.load(open(f"{FOLDER}/blip_laion_cc_sbu_558k.json", "r"))
    with open(DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]
    data = data
    print(len(data))
    with Pool(args.num_workers) as p:
        result = list(tqdm(p.imap(process_file, data), total=len(data)))

    result = [i for i in result if i is not None]
    print(len(result))
    # json.dump(
    #     result,
    #     open(f"{FOLDER}/{DATA_PATH}_cleaned.json", "w"),
    #     indent=4,
    # )


if __name__ == "__main__":
    main()
