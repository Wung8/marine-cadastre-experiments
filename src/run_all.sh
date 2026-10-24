#!/usr/bin/env bash
# Baselines, then the 8 (position encoding x history) runs, one at a time.
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-30000}
python src/train.py --baselines > logs/baselines.log 2>&1 && echo "done baselines" || echo "FAILED baselines"
for pe in fourier grid; do
  for h in 0 3 6 12; do
    if [ -f "results/${pe}_h${h}.json" ]; then echo "skip ${pe}_h${h}"; continue; fi
    if python src/train.py --posenc "$pe" --hist "$h" --steps "$STEPS" > "logs/train_${pe}_h${h}.log" 2>&1; then
      echo "done ${pe}_h${h}"
    else
      echo "FAILED ${pe}_h${h}"
    fi
  done
done
echo "ALL DONE"
