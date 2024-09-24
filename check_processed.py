import json
import os
import re

meta = "./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl"
# data = [json.loads(q) for q in open(meta, "r")]
# data = json.load(open(meta, "r"))
if meta.endswith(".jsonl"):
    data = [json.loads(q) for q in open(meta, "r")]
else:
    data = json.load(open(meta, "r"))
image_folder = "./playground/data/eval/gqa/data/images"
input_paths = []
for i in data:
    if "image" in i.keys():
        image_path = os.path.join(image_folder, i["image"])
        seg_path = re.sub(r"\.(jpg|jpeg|png|bmp|gif)$", ".npz", image_path)
        id_path = seg_path.replace(".npz", "_id.json")
        ids = json.load(open(id_path, "r"))
        if len(ids) == 0:
            print(image_path)
