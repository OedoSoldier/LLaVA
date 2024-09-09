#!/bin/bash

PYTHONPATH=~/workspace/LLaVA_obj/LLaVA python llava/eval/model_vqa_loader.py \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/wangsihan/workspace/data/LLaVA-Evaluation/coco/images \
    --answers-file ./playground/data/eval/pope/answers/llava-vicuna-7b-v1.5-finetune.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

PYTHONPATH=~/workspace/LLaVA_obj/LLaVA python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-vicuna-7b-v1.5-finetune.jsonl
