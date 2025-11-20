#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:27:54 2025

@author: isara
"""

import os
from sklearn.model_selection import train_test_split

def get_all_file_paths(target_dir):
    """
    Return a list of X,Y
    """    
    feature_paths = []
    label_paths = []
    for (root,_,files) in os.walk(target_dir):
        # print(root)
        for file in files:
            # print(file)
            f_path = os.path.join(root, file) # just string concat
            # l_path = os.path.join(root, )
            feature_paths.append(f_path)
    return feature_paths

# def get_all_file_paths(read_dir):
#     """
#     Return a list of absolute paths to all files in dir(ectory)
#     """
#     file_paths = []
#     for (root, _, files) in os.walk(read_dir):
#         for file in files:
#             abs_path = os.path.join(root, file) # just string concat
#             file_paths.append(abs_path)
#     return file_paths

dataset = "HyperSim"
datasubset = "ai_001_001"   # "ai_001_001", "ai_001_002", "split_test", "preview"            
target_dir = f"/projectnb/cs523aw/students/isara/software/DepthDatasets/{dataset}/all/{datasubset}"
all_data_paths = get_all_file_paths(target_dir)

for data in all_data_paths:
    print(data)

# dataset_text_path = f"/usr4/cs523aw/isara/depth_estimation/Depth-Anything-V2/metric_depth/dataset/splits/{dataset}/{datasubset}"
# train_filename = "train.txt"
# test_filename = "val.txt"
# train_txt_path = os.path.join(dataset_text_path, train_filename)
# test_txt_path = os.path.join(dataset_text_path, test_filename)

# save_all = False
# if save_all:
#     all_filename = "all.txt"
#     all_path = os.path.join(dataset_text_path, all_filename)

#     with open(all_path, 'w') as file:
#         for path in all_data_paths:
#             file.write(f"{path}\n") if i%2==1 else file.write(f"{path}\t")

# train_paths, test_paths = train_test_split(all_data_paths, test_size=0.2)

# with open(train_txt_path, 'w') as file:
#     for i, path in enumerate(train_paths):
#         file.write(f"{path}\n") if i%2==1 else file.write(f"{path}\t")
        
# with open(test_txt_path, 'w') as file:
#     for path in test_paths:
#         file.write(f"{path}\n") if i%2==1 else file.write(f"{path}\t")

# print(f"Train: {len(train_paths)} = {len(train_paths)/len(all_data_paths)*100:.2f}%")
# print(f"Test: {len(test_paths)} = {len(test_paths)/len(all_data_paths)*100:.2f}%")


