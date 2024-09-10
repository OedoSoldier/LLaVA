#!/bin/bash

# PYTHONPATH=~/workspace/LLaVA_obj/LLaVA nohup bash pretrain.sh > log.out 2>&1 &
# wait
PYTHONPATH=~/workspace/LLaVA_obj/LLaVA nohup bash finetune_lora.sh > 2.log.out 2>&1 &