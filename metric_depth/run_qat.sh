#!/bin/bash

encoder="vits"
max_depth=20
dataset="HyperSim"

# load_from=$model_path # $newpyt_model $oldpyt_model # finetune_39


python qat.py --encoder $encoder --max-depth $max_depth 2>&1 | tee -a "QAT/$(date +"%Y%m%d_%H%M%S")_batch4.log"

# --encoder vits --load-from "/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/PTQ/result_selected/torch.ao/128calibration/vits_HyperSim_pqt_newpyt_nonstrict_x86_v1.pth" --max-depth 20 --img-path "/home/tand/Documents/class/cs523/project/depth_anything_v2/HyperSim/all/ai_003_009/images/scene_cam_00_final_preview/frame.0000.color.jpg" --out-dir "depth_vis/ptq_newpyt"
