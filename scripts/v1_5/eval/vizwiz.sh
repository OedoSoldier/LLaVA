#!/bin/bash

python llava/eval/model_vqa_loader.py \
    --model-path checkpoints/llava-vicuna-13b-v1.5-finetune_dual_merged \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-vicuna-13b-v1.5-finetune_dual_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-vicuna-13b-v1.5-finetune_dual_merged.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-vicuna-13b-v1.5-finetune_dual_merged.json
