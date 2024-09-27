#!/bin/bash

python llava/eval/model_vqa.py \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_merged \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-vicuna-7b-v1.5-finetune_dual_merged.json

