#!/bin/bash

python llava/eval/model_vqa.py \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_merged \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/llava-vicuna-7b-v1.5-finetune_dual_merged2.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-vicuna-7b-v1.5-finetune_dual_merged2.jsonl
