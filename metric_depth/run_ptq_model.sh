#!/bin/bash

encoder="vits"

checkpoint="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
finetune_39="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/exp/HyperSim/ai_003_009/vits_HyperSim_v1.pth"
newpyt_model="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/PTQ/result_selected/torch.ao/128calibration/vits_HyperSim_pqt_newpyt_nonstrict_x86_v1.pth"
oldpyt_model="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/PTQ/result_selected/torch.ao/128calibration/vits_HyperSim_pqt_oldpyt_nonstrict_x86_v1.pth"

max_depth=20
img_path="/home/tand/Documents/class/cs523/project/depth_anything_v2//HyperSim/all/ai_003_009/images/scene_cam_01_final_preview/frame.0075.color.jpg"
out_dir="depth_vis/finetune_39" # finetune_39 ptq_newpyt ptq_oldpyt

load_from=$checkpoint # $newpyt_model $oldpyt_model # finetune_39


python run.py --encoder $encoder --load-from $load_from --max-depth $max_depth --img-path $img_path --outdir $out_dir #2>&1 | tee -a "profiler/$(date +"%Y%m%d_%H%M%S").log"

# --encoder vits --load-from "/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/PTQ/result_selected/torch.ao/128calibration/vits_HyperSim_pqt_newpyt_nonstrict_x86_v1.pth" --max-depth 20 --img-path "/home/tand/Documents/class/cs523/project/depth_anything_v2/HyperSim/all/ai_003_009/images/scene_cam_00_final_preview/frame.0000.color.jpg" --out-dir "depth_vis/ptq_newpyt"
