#!/usr/bin/env bash
# Heavy sub-13h campaign: Phase A (RAC definitive) then Phase B (chromatin healthy-prior training).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe"
LOG_TS() { date +"%Y-%m-%d %H:%M:%S"; }
echo "===== HEAVY CAMPAIGN START $(LOG_TS) ====="

echo "----- PHASE A: RAC definitive campaign $(LOG_TS) -----"
PYTHONUNBUFFERED=1 "$PY" -u scripts/heavy_rac_campaign.py
echo "----- PHASE A done rc=$? $(LOG_TS) -----"

echo "----- PHASE B: chromatin healthy-prior training $(LOG_TS) -----"
PYTHONUNBUFFERED=1 "$PY" -u -m pdac_circuit.pipeline.cli chromatin-train \
  --config configs/chromatin-heavy-campaign.json \
  --shards "data/processed/chromatin_encode_healthy_full_v1/*/*.npz" \
  --stage healthy_prior \
  --fasta data/raw/hg38-ref/hg38.fa \
  --device cuda \
  --checkpoint-dir models/chromatin/healthy_prior_ckpt \
  --allow-low-vram
echo "----- PHASE B done rc=$? $(LOG_TS) -----"

echo "----- extracting chromatin loss curve $(LOG_TS) -----"
"$PY" scripts/extract_train_curve.py models/chromatin/healthy_prior_ckpt results/chromatin_training_curve.json
echo "===== HEAVY CAMPAIGN END $(LOG_TS) ====="
