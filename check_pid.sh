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
NCCL_DEBUG=INFO nohup bash finetune_lora.sh > 2.log.out 2>&1 &
