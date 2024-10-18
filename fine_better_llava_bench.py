import json

# load jsonl
review1 = "/home/wangsihan/workspace/LLaVA_obj/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b-eval1.jsonl"
review1 = [json.loads(r) for r in open(review1, "r")]
review2 = "/home/wangsihan/workspace/LLaVA_obj/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews/llava-vicuna-13b-v1.5-finetune_dual_merged1.jsonl"
review2 = [json.loads(r) for r in open(review2, "r")]

for i in range(len(review1)):
    score1 = review1[i]["tuple"][1] / review1[i]["tuple"][0]
    score2 = review2[i]["tuple"][1] / review2[i]["tuple"][0]
    if score1 < score2:
        print(
            f"{review1[i]['answer2_id']}: {score1}, {review2[i]['answer2_id']}: {score2}"
        )
