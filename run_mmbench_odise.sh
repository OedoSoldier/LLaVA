#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run_mmbench_odise.py \
        --data-path ./playground/data/eval/mmbench/mmbench_dev_cn_20231003.tsv \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait