#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

python llava/eval/model_vqa_mmbench.py \
    --model-path checkpoints/llava-vicuna-13b-v1.5-finetune_dual_merged \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-vicuna-13b-v1.5-finetune_dual_merged.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-vicuna-13b-v1.5-finetune_dual_merged
