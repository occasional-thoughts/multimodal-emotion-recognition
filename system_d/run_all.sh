#!/bin/bash
# Runs every remaining config sequentially. Skips ones already finished,
# so it is safe to stop with Ctrl-C and re-run later.
#
#   ./run_all.sh              # seed 0 only
#   ./run_all.sh 0 1 2        # seeds 0,1,2
cd "$(dirname "$0")"
PY=../env/bin/python
SEEDS=${@:-0}
mkdir -p results logs

for seed in $SEEDS; do
  for fusion in cross self; do
    for feats in mfcc mfcc+gfcc; do
      tag="${fusion}_$(echo $feats | tr -d '+')_s${seed}"
      if [ -f "results/${tag}.json" ]; then
        echo "SKIP  ${tag} (already done)"
        continue
      fi
      echo "RUN   ${tag}  ($(date '+%H:%M'))"
      $PY train.py --fusion $fusion --features $feats --epochs 150 --seed $seed \
          > "logs/${tag}.log" 2>&1
      tail -2 "logs/${tag}.log" | head -1
    done
  done
done
echo; echo "ALL DONE. Summary:"; $PY analyze.py
