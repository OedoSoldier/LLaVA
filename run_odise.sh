#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run_odise.py \
        --data-path /home/wangsihan/workspace/LLaVA_obj/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
        --image-folder /home/wangsihan/workspace/LLaVA_obj/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait