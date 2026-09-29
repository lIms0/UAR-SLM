#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

mkdir -p ./outputs/VERITE/hash_dataset/logs
mkdir -p ./outputs/VERITE/hash_dataset/visual_cues
mkdir -p ./outputs/VERITE/hash_dataset/clip_scores

echo "=================================================="
echo "[Stage 1-1] Visual cue extraction with Qwen2.5-VL-3B"
echo "Start: $(date)"
echo "=================================================="

python ./scripts/verite/extract_visual_cues_qwen25vl_imageonly.py \
  2>&1 | tee ./outputs/VERITE/hash_dataset/logs/run_visual_cues_qwen25vl3b_$(date +%Y%m%d_%H%M%S).log

echo "=================================================="
echo "[Stage 1-2] CLIP image-claim score computation"
echo "Start: $(date)"
echo "=================================================="

python ./scripts/verite/compute_clip_scores.py \
  2>&1 | tee ./outputs/VERITE/hash_dataset/logs/run_clip_scores_$(date +%Y%m%d_%H%M%S).log

echo "=================================================="
echo "[Stage 1 done]"
echo "End: $(date)"
echo "=================================================="
