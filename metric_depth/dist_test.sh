#!/bin/bash

checkpoint=1
encoder="vits"
dataset="HyperSim"
subdataset="ai_003_009"
save_path="exp/$dataset/$subdataset/test"

if [[ $checkpoint == 0 ]]; then
    load_from="exp/$dataset/$subdataset/vits_HyperSim_v1.pth" # or "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
    log_name="$save_path/$(date +"%Y%m%d_%H%M%S").log"
else
    load_from="checkpoints/depth_anything_v2_metric_hypersim_$encoder.pth"
    log_name="$save_path/checkpoints_$(date +"%Y%m%d_%H%M%S").log"
fi

touch $log_name

python test.py \
       --encoder=$encoder \
       --load-from=$load_from \
       --save-path=$save_path --dataset=$dataset \
       --min-depth=0.001 --max-depth=20.0 2>&1 | tee -a $log_name
