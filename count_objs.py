import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import re
import numpy as np
import argparse
import random
import math

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

with open(args.data_path, "r") as f:
    data = json.load(f)


def count_img_obj(data):
    img_count = 0
    obj_count = 0
    for i in data:
        if "ids" in i:
            obj_count += len(i["ids"])
            img_count += 1
    return img_count, obj_count


print(len(data), count_img_obj(data))

random.seed(42)

for p in [0.5, 0.4, 0.3, 0.2, 0.1]:
    temp = random.sample(data, math.ceil(len(data) * p))
    print(len(temp), p, count_img_obj(temp))
