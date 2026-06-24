#!/bin/bash
# Evaluate a fine-tuned LoRA GeoChat model on UCMerced or AID test split.
#
# Usage:
#   bash scripts/eval_finetuned.sh ucmerced
#   bash scripts/eval_finetuned.sh aid

set -euo pipefail

DATASET=${1:-ucmerced}
BASE_MODEL="./checkpoints/GeoChat"

if [ "$DATASET" = "ucmerced" ]; then
    MODEL_PATH="./checkpoints/GeoChat-FT-UCMerced"
    IMAGE_FOLDER="./datasets/UCmerced/Images"
    QUESTION_FILE="./data/finetune/UCMerced_test.jsonl"
    ANSWERS_FILE="./results/ft_ucmerced.jsonl"
elif [ "$DATASET" = "aid" ]; then
    MODEL_PATH="./checkpoints/GeoChat-FT-AID"
    IMAGE_FOLDER="./datasets/AID"
    QUESTION_FILE="./data/finetune/AID_test.jsonl"
    ANSWERS_FILE="./results/ft_aid.jsonl"
else
    echo "Unknown dataset: $DATASET (use 'ucmerced' or 'aid')"
    exit 1
fi

echo "=== Evaluating fine-tuned GeoChat on $DATASET ==="
echo "  Model:    $MODEL_PATH"
echo "  Base:     $BASE_MODEL"
echo "  Images:   $IMAGE_FOLDER"
echo "  Questions: $QUESTION_FILE"
echo "  Output:   $ANSWERS_FILE"

python geochat/eval/batch_geochat_scene.py \
    --model-path "$MODEL_PATH" \
    --model-base "$BASE_MODEL" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode llava_v1 \
    --batch_size 1

echo "=== Evaluation complete: $ANSWERS_FILE ==="
