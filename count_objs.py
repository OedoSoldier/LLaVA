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
args = parser.parse_args()

with open(args.data_path, "r") as f:
    data = json.load(f)

obj_count = 0
for i in data:
    if "ids" in i:
        obj_count += len(i["ids"])

print(obj_count)
