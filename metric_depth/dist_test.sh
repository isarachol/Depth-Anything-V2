#!/bin/bash

python test.py \
       --encoder=vitb \
       --load-from=exp/HyperSim/ai_001_001/vitb_HyperSim_v1.pth \
       --save-path=exp/HyperSim/ai_001_001/test --dataset=HyperSim \
       --min-depth=0.001 --max-depth=20.0 2>&1 | tee -a exp/HyperSim/ai_001_001/test/$(date +"%Y%m%d_%H%M%S").log
