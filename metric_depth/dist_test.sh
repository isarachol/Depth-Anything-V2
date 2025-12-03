#!/bin/bash

checkpoint=0
local="local_" # or ""
encoder="vits"
dataset="HyperSim"
subdataset="ai_003_009" # ai_001_001, ai_003_009
save_path="test_log/$subdataset/test_quantized" # /test_$encoder

if [[ $checkpoint != 1 ]]; then
    #load_from="exp/$dataset/$subdataset/${encoder}_HyperSim_v1.pth" # vits_HyperSim_pqt_newpyt_nonstrict_x86_v1.pth vits_HyperSim_pqt_newpyt_nonstrict_xnn_v1.pth
    load_from="PTQ/result_selected/torch.ao/x86_vs_xnn/vits_HyperSim_pqt_oldpyt_nonstrict_x86_v1.pth"
    log_name="$save_path/${local}$(date +"%Y%m%d_%H%M%S").log"
else
    load_from="checkpoints/depth_anything_v2_metric_hypersim_$encoder.pth" 
    log_name="$save_path/${local}checkpoints_$(date +"%Y%m%d_%H%M%S").log"
fi

touch $log_name

python test.py \
       --encoder=$encoder \
       --load-from=$load_from \
       --save-path=$save_path --dataset=$dataset \
       --min-depth=0.001 --max-depth=20.0 2>&1 | tee -a $log_name
       
# --encoder "vits" --load-from "metric_depth/PTQ/result_selected/torch.ao/vits_HyperSim_pqt_oldpyt_nonstrict_x86_v1.pth" --save-path "metric_depth/test_log/ai_003_009/test_quantized" --dataset "HyperSim"
