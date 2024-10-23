#!/bin/bash

python llava/eval/model_vqa_loader.py \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_no_pad_merged \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-vicuna-7b-v1.5-finetune_dual_no_pad_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-vicuna-7b-v1.5-finetune_dual_no_pad_merged

cd eval_tool

python calculation.py --results_dir answers/llava-vicuna-7b-v1.5-finetune_dual_no_pad_merged
