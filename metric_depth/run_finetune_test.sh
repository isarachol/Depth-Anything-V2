#!bin/bash

python run.py --encoder vitb --load-from exp/HyperSim/ai_001_001/vitb_HyperSim_v1.pth --max-depth 20 --img-path /projectnb/cs523aw/students/isara/software/DepthDatasets/HyperSim/all/ai_001_001/images/scene_cam_00_final_preview/frame.0042.color.jpg --outdir depth_vis

