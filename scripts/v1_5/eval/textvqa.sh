#!/bin/bash

python llava/eval/model_vqa_loader.py \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_merged \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_textvqa.py \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-vicuna-7b-v1.5-finetune_dual_merged.jsonl
