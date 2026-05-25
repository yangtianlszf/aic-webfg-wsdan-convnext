#!/usr/bin/env bash
set -e

python src/wsdan_convnext_plus.py \
  --mode train \
  --train_dir data/WebiNat-5000/train \
  --val_dir data/WebiNat-5000/val \
  --outdir runs/web5000_convnext_base \
  --arch convnext_base \
  --img_size 384 \
  --batch_size 32 \
  --epochs 80 \
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
