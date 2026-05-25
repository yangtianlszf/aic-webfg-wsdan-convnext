#!/usr/bin/env bash
set -e

python src/wsdan_convnext_plus.py \
  --mode predict \
  --test_dir data/WebiNat-5000/test \
  --checkpoint runs/web5000_convnext_base/best.pt \
  --csv_path submissions/pred_results_web5000.csv \
  --img_size 384 \
  --batch_size 64 \
  --workers 8 \
  --tta_flip
