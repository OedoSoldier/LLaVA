#!/bin/bash

python llava/eval/model_vqa_science \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_merged \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-vicuna-7b-v1.5-finetune_dual_merged_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-vicuna-7b-v1.5-finetune_dual_merged_result.json
