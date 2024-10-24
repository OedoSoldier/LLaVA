#!/bin/bash

# nohup bash pretrain.sh > log.out 2>&1 &
# wait
NCCL_DEBUG=INFO nohup bash finetune_lora.sh > 4.log.out 2>&1 &