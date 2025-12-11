#!/bin/bash

encoder="vits"
max_depth=20
out_dir="demo" # combined
img_path="/home/tand/Pictures/Webcam/2025-12-11-141110.jpg"
# /home/tand/Pictures/Webcam/2025-12-11-141110.jpg
# assets/examples/demo03.jpg

load_from="QAT/int8_group128/vits_HyperSim_quantized_int8_v2.pth"
# load_from="QAT/int8_group32/vits_HyperSim_quantized_int8_v1.pth"
# leave blank for baseline

python demo.py --encoder $encoder --max-depth $max_depth --outdir $out_dir --img-path $img_path #--load-from $load_from 
