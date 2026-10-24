#!/usr/bin/env bash
# Noise ablation -> data age -> drift embedding -> noise+drift 2x2. Resumable.
set -u
cd "$(dirname "$0")/.."
for phase in noise age drift factorial; do
  echo "phase $phase start $(date +%H:%M)"
  if python src/drift.py --phase "$phase" --seeds 0 1 2 > "logs/drift_${phase}.log" 2>&1; then
    echo "phase $phase done $(date +%H:%M)"
  else
    echo "FAILED phase $phase $(date +%H:%M)"
  fi
done
echo "ALL DONE $(date +%H:%M)"
