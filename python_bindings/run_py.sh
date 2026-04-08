#!/bin/bash
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

python3 infer.py \
  --model ../config/DStereoV2.4_int16_uncertainty.bin \
  --left ../standalone/img/2/left000001.png \
  --right ../standalone/img/2/right000001.png \
  --out_dir ./result \
  --uncertainty_th 0.0 \
  --fx 300.0 \
  --baseline 0.06 \
  --doffs 0.0 \
  --save_vis
