#!/bin/bash

checkpoint=1
encoder="vits"
dataset="HyperSim"
subdataset="ai_003_009" # ai_001_001, ai_003_009
save_path="test_log/$subdataset/test_$encoder"

if [[ $checkpoint != 1 ]]; then
    load_from="exp/$dataset/$subdataset/${encoder}_HyperSim_v1.pth"
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
