#!/bin/bash

quantizer="x86" # x86 xnn --> bad
strict="nonstrict" # strict
test_lim=1 # None or number
encoder="vits"
load_from="/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/exp/HyperSim/ai_003_009/vits_HyperSim_v1.pth"
max_depth=20
pyt_ver="newpyt" # oldpyt
out_dir="ptq_log_${pyt_ver}"
dataset="HyperSim"
filename="$out_dir/$(date +"%Y%m%d_%H%M%S")_${strict}_${quantizer}_${pyt_ver}.log"


touch $filename

python run_ptq_${quantizer}.py --dataset $dataset --encoder $encoder --load-from $load_from --max-depth $max_depth --outdir $out_dir --test-lim $test_lim --strict $strict --pyt-ver $pyt_ver 2>&1 | tee -a $filename

# --dataset HyperSim --encoder vits --load-from /home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/exp/HyperSim/ai_003_009/vits_HyperSim_v1.pth --max-depth 20 --outdir ptq_log_oldpyt --strict 'nonstrict' --pyt-ver 'oldpyt' --test-lim 2
