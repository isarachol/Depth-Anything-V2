#!/bin/bash

# checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \

# for debugging
# python run.py \
#   --encoder vitb --load-from metric_depth/exp/HyperSim/ai_001_001/vitb_HyperSim_v1.pth --max-depth 20 --img-path metric_depth/assets/examples/demo01.jpg --outdir metric_depth/depth_vis

# finetune1
python run.py \
  --encoder vitb --load-from exp/HyperSim/ai_001_001/vitb_HyperSim_v1.pth --max-depth 20 --img-path assets/examples/demo10.jpg --outdir depth_vis/finetuned1

# vitb
# python run.py \
#   --encoder vitb --load-from checkpoints/depth_anything_v2_metric_hypersim_vitb.pth --max-depth 20 --img-path assets/examples/demo10.jpg --outdir depth_vis/vitb

# vits
# python run.py \
#   --encoder vits --load-from checkpoints/depth_anything_v2_metric_hypersim_vits.pth --max-depth 20 --img-path assets/examples/demo10.jpg --outdir depth_vis/vits