import json
import glob
import random

random.seed(42)

FOLDER = "/home/wangsihan/workspace/data/LLaVA-IOT-Finetune"
files = glob.glob(f"{FOLDER}/processed/*.json")
data = []

for file in files:
    data += json.load(open(file, "r"))

print(len(data))

# shuffle data
random.shuffle(data)

for idx, i in enumerate(data):
    i["id"] = idx

json.dump(
    data,
    open(f"{FOLDER}/llava_iot_scaling.json", "w"),
    indent=4,
)
