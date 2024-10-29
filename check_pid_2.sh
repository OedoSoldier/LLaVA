#!/bin/bash

PID=3734604
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running"
    sleep 10
done
 
# 倒计时结束后执行的操作
echo "Process $PID has finished"

source ~/workspace/miniconda3/etc/profile.d/conda.sh
conda activate llava
# NCCL_DEBUG=INFO nohup bash finetune_lora.sh > 2.log.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/merge_lora_weights.py --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_lora_20_data/ --model-base checkpoints/vicuna-7b-v1.5/ --save-model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_20_data_merged/

CUDA_VISIBLE_DEVICES=1 python llava/eval/model_vqa_loader.py \
    --model-path checkpoints/llava-vicuna-7b-v1.5-finetune_dual_20_data_merged \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-vicuna-7b-v1.5-finetune_dual_20_data_merged.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
