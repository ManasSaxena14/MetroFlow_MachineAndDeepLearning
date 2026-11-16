#!/usr/bin/env bash
# Phase 5: every graph-model run reported in docs/phase_reports/PHASE_5_REPORT.md.
# Main models: 3 seeds each. Ablations: seed 0, same budget as the main run they are compared with.
# Runs that already have results/graph/<run>.csv are skipped, so the script can be stopped and resumed.
#
# Usage: scripts/run_phase5.sh [stgat|dcrnn|all]     (default all = the two queues side by side on one GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u -W ignore -m src.train.graph"
STGAT="--epochs 40 --patience 8 --batch 32 --lr 0.001 --skip-done"
DCRNN="--epochs 30 --patience 8 --batch 64 --skip-done"
mkdir -p logs

stgat_queue() {
  for s in 0 1 2; do $PY stgat --seed $s $STGAT; done
  for g in identity forward backward undirected random; do   # what graph does STGAT need?
    $PY stgat --graph $g --seed 0 $STGAT
  done
  $PY stgat --gat-layers 1 --seed 0 $STGAT                     # hops: 1 vs 2 attention layers
  $PY stgat --dropout 0.0 --seed 0 $STGAT                      # regularisation: dropout off
}

dcrnn_queue() {
  for s in 0 1 2; do $PY dcrnn --seed $s $DCRNN; done
  $PY dcrnn --k 1 --seed 0 $DCRNN                               # hops: K = 1 vs 2 diffusion steps
  $PY dcrnn --supports forward --seed 0 $DCRNN                  # one-way vs dual random walk
  # reference window split, to compare with the published METR-LA numbers
  [[ -f data/processed/dcrnn/train.npz ]] || .venv/bin/python -m src.data.preprocess --protocol dcrnn
  $PY dcrnn --protocol dcrnn --seed 0 $DCRNN
}

case "${1:-all}" in
  stgat) stgat_queue ;;
  dcrnn) dcrnn_queue ;;
  all)   stgat_queue > logs/phase5_stgat.log 2>&1 &
         dcrnn_queue > logs/phase5_dcrnn.log 2>&1 &
         wait ;;
esac
$PY --table > /dev/null
