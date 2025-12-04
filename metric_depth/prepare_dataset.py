#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:27:54 2025

@author: isara
"""

import os
from sklearn.model_selection import train_test_split

def get_all_file_paths(feature_dir, label_dir, label_type):
    """
    Return a list of pair of (X,Y)
    """    
    data_paths = []
    for (_,_,files) in os.walk(feature_dir):
        for file in files:
            if 'tonemap' in file: # only use tonemap images
                x_path = os.path.join(feature_dir, file) # just string concat
                label_file = file[0:11]+label_type # frame.####.(label_type)
                y_path = os.path.join(label_dir, label_file)
                if os.path.exists(y_path):
                    line_path = x_path + " " + y_path
                    data_paths.append(line_path)
                else:
                    print("y_path doesn't exist!")
    return data_paths

dataset = "HyperSim"
datasubset = "ai_003_009"   # "ai_001_001", "ai_001_002", "split_test", "ai_003_009"

# Find images
local_decom_dir = "/home/tand/Documents/class/cs523/project/depth_anything_v2/" 
scc_decom_dir = "/projectnb/cs523aw/students/isara/software/DepthDatasets/"

dataset_dir = f"{local_decom_dir}/{dataset}/all/{datasubset}/images" #f"/projectnb/cs523aw/students/isara/software/DepthDatasets/{dataset}/all/{datasubset}/images"
feature_dir = f"{dataset_dir}/scene_cam_00_final_preview" # scene_cam_00_final_preview and scene_cam_01_final_preview
label_dir = f"{dataset_dir}/scene_cam_00_geometry_hdf5" # scene_cam_00_geometry_hdf5 and scene_cam_01_geometry_hdf5
label_type = "depth_meters.hdf5"

all_dataset1 = get_all_file_paths(feature_dir, label_dir, label_type)

feature_dir2 = f"{dataset_dir}/scene_cam_01_final_preview"
label_dir2 = f"{dataset_dir}/scene_cam_01_geometry_hdf5"
all_dataset2 = get_all_file_paths(feature_dir2, label_dir2, label_type)

all_dataset = all_dataset1 + all_dataset2

# Save directories
local_work_dir = "/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/"
scc_work_dir = "/usr4/cs523aw/isara/depth_estimation/Depth-Anything-V2/"

save_dir = f"{local_work_dir}metric_depth/dataset/splits/{dataset}/{datasubset}"

train_filename = "train.txt"
val_filename = "val.txt"
test_filename = "test.txt"
train_path = os.path.join(save_dir, train_filename)
val_path = os.path.join(save_dir, val_filename)
test_path = os.path.join(save_dir, test_filename)

save_all = False
if save_all:
    all_filename = "all.txt"
    all_path = os.path.join(save_dir, all_filename)

    with open(all_path, 'w') as file:
        for path in all_dataset:
            file.write(f"{path}\n")

# split data into train, val, test (70, 10, 20)
train_set, test_set = train_test_split(all_dataset, test_size=0.2) #test 20%
train_set, val_set = train_test_split(train_set, test_size=0.125) #val 10%

# save them
with open(train_path, 'w') as file:
    for path in train_set:
        file.write(f"{path}\n")

with open(val_path, 'w') as file:
    for path in val_set:
        file.write(f"{path}\n")

with open(test_path, 'w') as file:
    for path in test_set:
        file.write(f"{path}\n")

print(f'{all_dataset[len(all_dataset)-1]}')
print(f"All: {len(all_dataset)}")
print(f"Train: {len(train_set)}")
print(f"Val: {len(val_set)}")
print(f"Test: {len(test_set)}")
