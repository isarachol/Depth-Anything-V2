#!/bin/bash

encoder="vits"
model="quantized_v2" # checkpoint finetuned_v1 finetuned_v2 quantized_v1 quantized_v2

if [ $model == "checkpoint" ]; then
	model_path="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
elif [ $model == "finetuned_v1" ]; then
	model_path="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/QAT/int8_group32/vits_HyperSim_finetuned_v1.pth"
elif [ $model == "finetuned_v2" ]; then
	model_path="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/QAT/int8_group128/vits_HyperSim_finetuned_v2.pth"
elif [ $model == "quantized_v1" ]; then
	model_path="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/QAT/int8_group32/vits_HyperSim_quantized_int8_v1.pth"
elif [ $model == "quantized_v2" ]; then
	model_path="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/QAT/int8_group128/vits_HyperSim_quantized_int8_v2.pth"
else
	echo "wrong model name"
	exit 1
fi

max_depth=20
img_path="/home/tand/Documents/class/cs523/project/depth_anything_v2//HyperSim/all/ai_003_009/images/scene_cam_01_final_preview/frame.0075.color.jpg"
dataset="HyperSim"
out_dir="depth_vis/qat/${model}"

load_from=$model_path # $newpyt_model $oldpyt_model # finetune_39


python qat_make_figs.py --encoder $encoder --load-from $load_from --max-depth $max_depth --dataset $dataset --outdir $out_dir #2>&1 | tee -a "profiler/$(date +"%Y%m%d_%H%M%S").log" --img-path $img_path

# --encoder vits --load-from "/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/PTQ/result_selected/torch.ao/128calibration/vits_HyperSim_pqt_newpyt_nonstrict_x86_v1.pth" --max-depth 20 --img-path "/home/tand/Documents/class/cs523/project/depth_anything_v2/HyperSim/all/ai_003_009/images/scene_cam_00_final_preview/frame.0000.color.jpg" --out-dir "depth_vis/ptq_newpyt"
