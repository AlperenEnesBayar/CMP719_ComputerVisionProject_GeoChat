#!/bin/bash
# Full fine-tuning pipeline: UCMerced → AID (sequential, single GPU)
# Estimated total time: ~2.5 hours
set -euo pipefail

cd "$(dirname "$0")/.."

echo "======================================================"
echo " Step 1/4: Fine-tuning GeoChat on UCMerced"
echo "======================================================"
bash scripts/finetune_lora_single_gpu.sh ucmerced

echo "======================================================"
echo " Step 2/4: Evaluating fine-tuned model on UCMerced"
echo "======================================================"
bash scripts/eval_finetuned.sh ucmerced

echo "======================================================"
echo " Step 2b: GeoAgent eval on fine-tuned UCMerced model"
echo "======================================================"
python geochat/eval/eval_geoagent_scene_simple.py \
    --model-path ./checkpoints/GeoChat-FT-UCMerced \
    --model-base ./checkpoints/GeoChat \
    --image-folder ./datasets/UCmerced/Images \
    --question-file ./data/finetune/UCMerced_test.jsonl \
    --answers-file ./results/ft_agent_ucmerced.jsonl \
    --conv-mode llava_v1

echo "======================================================"
echo " Step 3/4: Fine-tuning GeoChat on AID"
echo "======================================================"
bash scripts/finetune_lora_single_gpu.sh aid

echo "======================================================"
echo " Step 4/4: Evaluating fine-tuned model on AID"
echo "======================================================"
bash scripts/eval_finetuned.sh aid

echo "======================================================"
echo " Step 4b: GeoAgent eval on fine-tuned AID model"
echo "======================================================"
python geochat/eval/eval_geoagent_scene_simple.py \
    --model-path ./checkpoints/GeoChat-FT-AID \
    --model-base ./checkpoints/GeoChat \
    --image-folder ./datasets/AID \
    --question-file ./data/finetune/AID_test.jsonl \
    --answers-file ./results/ft_agent_aid.jsonl \
    --conv-mode llava_v1

echo "======================================================"
echo " All done! Results:"
echo "  UCMerced FT:       results/ft_ucmerced.jsonl"
echo "  UCMerced FT+Agent: results/ft_agent_ucmerced.jsonl"
echo "  AID FT:            results/ft_aid.jsonl"
echo "  AID FT+Agent:      results/ft_agent_aid.jsonl"
echo "======================================================"
