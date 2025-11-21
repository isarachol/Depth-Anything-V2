#!/bin/bash -l

#$ -P cs523aw
#$ -o exp/HyperSim/hypersim1_1_vits_sptest/finetune_out
#$ -e exp/HyperSim/hypersim1_1_vits_sptest/finetune_error
#$ -M isara@bu.edu
#$ -m beas
#$ -l mem_per_core=16G
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l gpu_memory=16G


# prepare environment
module load miniconda
module load academic-ml/fall-2025

conda activate /projectnb/cs523aw/students/isara/software/dav2

# code to run
now=$(date +"%Y%m%d_%H%M%S")

epoch=120
bs=4
gpus=1
lr=0.000005
encoder=vits
dataset=HyperSim # hypersim vkitti
img_size=518
min_depth=0.001
max_depth=20 # 80 for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/HyperSim # exp/vkitti

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
