#!/bin/bash
# Single-GPU LoRA fine-tuning for scene classification datasets.
# Requires ~12 GB VRAM (RTX 4060 Ti / 3080 / 4080 etc.)
#
# Usage:
#   bash scripts/finetune_lora_single_gpu.sh ucmerced
#   bash scripts/finetune_lora_single_gpu.sh aid
#
# Outputs:
#   checkpoints/GeoChat-FT-UCMerced/
#   checkpoints/GeoChat-FT-AID/

set -euo pipefail

DATASET=${1:-ucmerced}
MODEL_PATH="./checkpoints/GeoChat"

if [ "$DATASET" = "ucmerced" ]; then
    DATA_JSON="./data/finetune/UCMerced_train.json"
    IMAGE_FOLDER="./datasets/UCmerced/Images"
    OUTPUT_DIR="./checkpoints/GeoChat-FT-UCMerced"
elif [ "$DATASET" = "aid" ]; then
    DATA_JSON="./data/finetune/AID_train.json"
    IMAGE_FOLDER="./datasets/AID"
    OUTPUT_DIR="./checkpoints/GeoChat-FT-AID"
else
    echo "Unknown dataset: $DATASET (use 'ucmerced' or 'aid')"
    exit 1
fi

echo "=== Fine-tuning GeoChat on $DATASET ==="
echo "  Data:   $DATA_JSON"
echo "  Images: $IMAGE_FOLDER"
echo "  Output: $OUTPUT_DIR"

WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python geochat/train/train_mem.py \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --model_name_or_path "$MODEL_PATH" \
    --version v1 \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 False \
    --fp16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 2

echo "=== Fine-tuning complete. Checkpoint: $OUTPUT_DIR ==="
