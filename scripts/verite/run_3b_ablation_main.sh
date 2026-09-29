#!/usr/bin/env bash
set -e

cd /home/oem/UAR-SLM

FEATURE_PATH="data/VERITE/processed/full_usable/splits/verite_stage1_features_test.jsonl"
EVIDENCE_PATH="data/VERITE/processed/full_usable/splits/retrieved_evidence_brave_q4_k10_test.jsonl"
MODEL_TAG="llama32_3b"
MAX_NEW_TOKENS=24

LOG_DIR="outputs/VERITE/full_usable/main_comparison/run_logs"
mkdir -p "$LOG_DIR"

run_if_missing() {
  local out_file="$1"
  shift

  if [ -f "$out_file" ]; then
    echo "[SKIP] already exists: $out_file"
  else
    echo "[RUN] $out_file"
    "$@"
  fi
}

echo "===== Ablation 1: Always-RAG evidence count k ====="

for K in 1 3 5 10; do
  RUN_NAME="llama32_3b_always_rag_test_k${K}"
  OUT_FILE="outputs/VERITE/full_usable/main_comparison/always_rag/${RUN_NAME}.jsonl"

  run_if_missing "$OUT_FILE" \
    python scripts/verite/run_always_rag.py \
      --feature_path "$FEATURE_PATH" \
      --evidence_path "$EVIDENCE_PATH" \
      --model_tag "$MODEL_TAG" \
      --run_name "$RUN_NAME" \
      --max_evidence "$K" \
      --max_new_tokens "$MAX_NEW_TOKENS"
done

echo "===== Ablation 2: Always-GraphRAG evidence count k ====="

for K in 1 3 5 10; do
  RUN_NAME="llama32_3b_always_graphrag_test_k${K}_edge2"
  OUT_FILE="outputs/VERITE/full_usable/main_comparison/always_graphrag/${RUN_NAME}.jsonl"

  run_if_missing "$OUT_FILE" \
    python scripts/verite/run_always_graphrag.py \
      --feature_path "$FEATURE_PATH" \
      --evidence_path "$EVIDENCE_PATH" \
      --model_tag "$MODEL_TAG" \
      --run_name "$RUN_NAME" \
      --max_evidence "$K" \
      --edge_min_overlap 2 \
      --max_new_tokens "$MAX_NEW_TOKENS"
done

echo "===== Ablation 3: Always-GraphRAG edge threshold ====="

for EDGE in 1 2 3; do
  RUN_NAME="llama32_3b_always_graphrag_test_k5_edge${EDGE}"
  OUT_FILE="outputs/VERITE/full_usable/main_comparison/always_graphrag/${RUN_NAME}.jsonl"

  run_if_missing "$OUT_FILE" \
    python scripts/verite/run_always_graphrag.py \
      --feature_path "$FEATURE_PATH" \
      --evidence_path "$EVIDENCE_PATH" \
      --model_tag "$MODEL_TAG" \
      --run_name "$RUN_NAME" \
      --max_evidence 5 \
      --edge_min_overlap "$EDGE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
done

echo "===== Ablation runs done ====="
