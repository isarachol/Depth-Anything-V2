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
datasubset = "ai_001_001"   # "ai_001_001", "ai_001_002", "split_test"           
dataset_dir = f"/projectnb/cs523aw/students/isara/software/DepthDatasets/{dataset}/all/{datasubset}/images"
feature_dir = f"{dataset_dir}/scene_cam_00_final_preview"
label_dir = f"{dataset_dir}/scene_cam_00_geometry_hdf5"
label_type = "depth_meters.hdf5"

all_dataset = get_all_file_paths(feature_dir, label_dir, label_type)

save_dir = f"/usr4/cs523aw/isara/depth_estimation/Depth-Anything-V2/metric_depth/dataset/splits/{dataset}/{datasubset}"
train_filename = "train.txt"
test_filename = "val.txt"
train_path = os.path.join(save_dir, train_filename)
test_path = os.path.join(save_dir, test_filename)

save_all = False
if save_all:
    all_filename = "all.txt"
    all_path = os.path.join(save_dir, all_filename)

    with open(all_path, 'w') as file:
        for path in all_dataset:
            file.write(f"{path}\n")

train_set, test_set = train_test_split(all_dataset, test_size=0.2)

with open(train_path, 'w') as file:
    for path in train_set:
        file.write(f"{path}\n")
        
with open(test_path, 'w') as file:
    for path in test_set:
        file.write(f"{path}\n")

# print(f"Train: {len(train_paths)} = {len(train_paths)/len(all_data_paths)*100:.2f}%")
# print(f"Test: {len(test_paths)} = {len(test_paths)/len(all_data_paths)*100:.2f}%")


