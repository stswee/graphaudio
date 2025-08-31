#!/usr/bin/env bash
set -euo pipefail
LOG_DIR=logs
mkdir -p "$LOG_DIR"
declare -a commands=(
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioBERT_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioBERT_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_BioClinicalBERT_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ text\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ mel\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GCN\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GAT\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k5.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
  CUDA_VISIBLE_DEVICES=0\ python\ train_embeddings.py\ --arch\ GraphSAGE\ --modality\ both\ --graph_file\ patient_graph/patient_graph_mel_generation_mel_k10.pt\ --text_dir\ embeddings/BioClinicalBERT_embeddings/\ --mel_dir\ mel_generation/\ --label_csv\ patient/patient_labels.csv
)
get_flag() {
  # Extract value following a named flag (e.g., --arch VALUE)
  # Usage: get_flag "<command string>" --arch
  local s="$1" flag="$2"
  awk -v f="$flag" '{
    for (i=1;i<=NF;i++) if ($i==f && i+1<=NF) { print $(i+1); exit }
  }' <<< "$s"
}

sanitize() {
  # Filename-safe slug
  sed -E 's/[^A-Za-z0-9._-]+/_/g' | sed -E 's/_+/_/g' | sed -E 's/^_+|_+$//g'
}

i=0
for cmd in "${commands[@]}"; do
  start_ts=$(date '+%Y-%m-%d %H:%M:%S')

  arch=$(get_flag "$cmd" --arch)
  modality=$(get_flag "$cmd" --modality)
  classification=$(get_flag "$cmd" --classification)  # may be empty if not provided
  graph_file=$(get_flag "$cmd" --graph_file)
  text_dir=$(get_flag "$cmd" --text_dir)
  clinical_csv=$(get_flag "$cmd" --clinical_csv)

  tag="$(printf '%s__%s' "${arch:-na}" "${modality:-na}")"
  if [ -n "${classification:-}" ]; then
    tag="${tag}__${classification}"
  fi

  gf_base=$(basename "${graph_file:-gf}")
  tag="${tag}__${gf_base}"

  # Prefer text_dir if present; otherwise include clinical_csv if present
  if [ -n "${text_dir:-}" ]; then
    td_base=$(basename "$text_dir")
    tag="${tag}__${td_base}"
  elif [ -n "${clinical_csv:-}" ]; then
    cc_base=$(basename "$clinical_csv")
    tag="${tag}__${cc_base}"
  fi

  safe_tag=$(printf '%s' "$tag" | sanitize)
  log_file="$LOG_DIR/${i}_${safe_tag}.log"

  echo "=====================================================" | tee -a "$log_file"
  echo "[START] $start_ts" | tee -a "$log_file"
  echo "[CMD] $cmd" | tee -a "$log_file"
  SECONDS=0

  # Run command (stdout+stderr -> log)
  set +e
  eval "$cmd" >>"$log_file" 2>&1
  exit_code=$?
  set -e

  duration=$SECONDS
  end_ts=$(date '+%Y-%m-%d %H:%M:%S')

  echo "[END] $end_ts" | tee -a "$log_file"
  echo "[DURATION] ${duration}s" | tee -a "$log_file"
  echo "[EXIT_CODE] ${exit_code}" | tee -a "$log_file"
  echo "=====================================================" | tee -a "$log_file"

  # If a job fails, continue to the next but note the nonzero exit code
  i=$((i+1))
done

echo "All experiments complete. Logs in $LOG_DIR"
