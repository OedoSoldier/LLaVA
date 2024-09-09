#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

MODEL_VERSION=vicuna-7b-v1.5
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

PYTHONPATH=~/workspace/LLaVA_obj/LLaVA nohup deepspeed --include localhost:2 llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ../../data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_cleaned.json \
    --image_folder ../../data/LLaVA-Pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --dual True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain_dual \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb > log.out 2>&1 & # --include localhost:0