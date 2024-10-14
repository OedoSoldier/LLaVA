#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run_odise.py \
        --data-path /data/wangsihan/data/LLaVA-IOT-Finetune/train_json/lvis_tune_220k_.json \
        --image-folder /data/wangsihan/data/LLaVA-IOT-Finetune \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait