#!/usr/bin/env bash
# Overnight re-run of all core models, chronological, single documented run.
# Sequential (RAM-safe, unattended). Per-model device chosen for speed:
#   GCN / SAGE / snapshot-temporal -> GPU (venv_gpu)
#   GAT / TGN                      -> CPU (venv_new)  [Python/attention-bound]
# Headline models (GCN, TGN) run early so they finish first.
# Fault-tolerant: a failed step is logged and the batch continues.
# Robustness (seed/sensitivity) is intentionally NOT here — run supervised later.

cd "D:/project-space/MDDB/masters/aml-gnn-deep" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU="venv_gpu/Scripts/python.exe"
CPU="venv_new/Scripts/python.exe"
MASTER="results/logs/overnight_master.log"
mkdir -p results/logs
echo "=========== OVERNIGHT RUN START $(date) ===========" | tee -a "$MASTER"

# Train every model fresh (snapshot trainer auto-resumes from checkpoints).
if ls results/checkpoints/*.pt >/dev/null 2>&1; then
  mkdir -p results/checkpoints/_prev
  mv results/checkpoints/*.pt results/checkpoints/_prev/ 2>/dev/null
  echo "Backed up stale checkpoints to _prev/" | tee -a "$MASTER"
fi

step () {
  local name="$1"; shift
  echo "----- [$name] START $(date) -----" | tee -a "$MASTER"
  if "$@"; then echo "----- [$name] OK $(date) -----" | tee -a "$MASTER"
  else echo "----- [$name] FAILED (exit $?) $(date) -----" | tee -a "$MASTER"; fi
}

# Tier 1 (fast, sklearn/xgboost) — re-run for single-environment consistency
step baselines $GPU experiments/run_baselines.py --variant HI-Small --seed 42
step ablation  $GPU experiments/run_ablation.py  --variant HI-Small --seed 42

# Headline static + headline temporal first
step gcn $GPU experiments/run_gnn.py --model gcn --device cuda --epochs 100
step tgn $CPU experiments/run_tgn.py --device cpu --epochs 100 --lr 0.003 \
    --pos_weight_mult 0.01 --grad_clip 0 --memory_dim 64 --time_dim 8

# Remaining static
step sage $GPU experiments/run_gnn.py --model sage --device cuda --epochs 100

# Snapshot temporal (weakest tier; lower cap, still early-stopped)
step temporal_gcn $GPU experiments/run_temporal.py --model temporal_gcn --device cuda --epochs 60
step evolve_gcn_h $GPU experiments/run_temporal.py --model evolve_gcn_h --device cuda --epochs 60 --rank 2

# GAT last (CPU; full-graph attention OOMs on 4GB GPU) — least central model
step gat $CPU experiments/run_gnn.py --model gat --device cpu --epochs 100 --heads 1

echo "=========== OVERNIGHT RUN DONE $(date) ===========" | tee -a "$MASTER"