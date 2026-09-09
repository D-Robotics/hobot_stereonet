#!/bin/bash
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

# Usage (same as ./infer): python3 infer.py [model_path] [local_img_dir] [uncertainty_th]
# The image dir is processed recursively; each scene dir needs camera_intrinsic.txt
# (or K.txt) and a left/right image pair.
python3 infer.py \
  ../config/DStereoV2.4_int16_uncertainty.bin \
  ../standalone/img \
  0.0 \
  --out_dir ./result
