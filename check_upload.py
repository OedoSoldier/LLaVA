import json

ori_data = json.load(
    open(
        "/home/wangsihan/workspace/LLaVA_obj/LLaVA/playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b.json",
        "r",
    )
)
my_data = json.load(
    open(
        "/home/wangsihan/workspace/LLaVA_obj/LLaVA/playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/llava-vicuna-7b-v1.5-finetune.json",
        "r",
    )
)

count = 0
total = 0
for i in range(len(ori_data)):
    if ori_data[i]["answer"] != "":
        total += 1
        if my_data[i]["answer"] == "":
            print(ori_data[i]["question_id"])
            count += 1

print(count, total)
