#!/usr/bin/env bash
set -e

python src/wsdan_convnext_plus.py \
  --mode train \
  --train_dir data/WebFG-400/train \
  --val_dir data/WebFG-400/val \
  --outdir runs/web400_convnext_base \
  --arch convnext_base \
  --img_size 384 \
  --batch_size 32 \
  --epochs 100 \
  --lr 5e-4 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --pretrained \
  --use_wsdan \
  --arcface \
  --loss elrplus \
  --amp \
  --class_balanced \
  --class_aware \
  --self_clean \
  --consistency_lambda 1.0
