#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################

deepspeed --include localhost:1,2 llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 --bbox_projector_lr 2e-5 \
    --model_name_or_path checkpoints/llava-$MODEL_VERSION-finetune_dual_merged \
    --version $PROMPT_VERSION \
    --data_path ../../data/LLaVA-IOT-Finetune/llava_iot_scaling_mix968k.json \
    --image_folder ../../data/LLaVA-IOT-Finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-finetune_dual_lora/non_lora_trainables.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --dual True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune_dual_stage3_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard # ./checkpoints/$MODEL_VERSION