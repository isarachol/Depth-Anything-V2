#!/bin/bash

encoder="vits"
load_from="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/exp/HyperSim/ai_003_009/vits_HyperSim_v1.pth"
max_depth=20
img_path="/home/tand/Documents/class/cs523/project/depth_anything_v2/HyperSim/all/ai_003_009/images/scene_cam_00_final_preview/frame.0000.color.jpg"
out_dir="depth_vis/finetuned2_vits"


python run.py --encoder $encoder --load-from $load_from --max-depth $max_depth --img-path $img_path --outdir $out_dir 2>&1 | tee -a "profiler/$(date +"%Y%m%d_%H%M%S").log"

