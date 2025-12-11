# Quantization-Aware Training for Depth Anything V2

This project is part of CS523 (Deep Learning) at BU. In this work, I explored and applied a deep neural network optimization technique called quantization-aware training, which shrinks the size of the neural network by changing variable types for the weight values. The result shows a decrease in weight size, a decrease in depth estimation performance, and an increase in inference time. Note that this is my first deep learning project, and some codes can be incorectly implemented. Please use the code at your own risk. 

To replicate the experiment, please see below.


## Setup
- Install dependencies for Depth Anything V2 by running `pip install -r requirements.txt`. For more information, please visit the official GitHub at https://github.com/DepthAnything/Depth-Anything-V2.git
- Install torchao `pip install torchao`
- download checkpoints
  - Relative depth small, used for finetuning: [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)
  - Metric depth small (Hypersim, indoor), used as baseline: [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true)
- dowlload the dataset I used ([Hypersim](https://github.com/apple/ml-hypersim/tree/main)), edit `code/python/tools/dataset_download_images.py` so that the only dataset is "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_009.zip". Or just [download](https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_009.zip). Set up the directory for the dataset, and modify the code `metric_depth/prepare_dataset.py` to create `.txt` files containing lists of paths to each image of training/validating/testing datasets

## Main scripts
- For training QAT, use the code `qat.py` by running `./run_qat.sh`. Note that some modifications, such as file paths need to be changed
- For inference only, use the code `demo.py` by running `./run_demo.sh`. Note that some modifications, such as file paths need to be changed

## Metric Depth
All the codes I developed are in the directory called metric_depth. I will mention some directories that are contributed by me and are relevant to this project.
- QAT: Logs for 4 types of QAT during training
- dataset/splits/HyperSim: contains `.txt` files with lists of paths to pictures used for training, validating, and testing (created by `prepare_dataset.py`). The only subset of data used is in `ai_003_009`, which is an indoor setting in as office. I chose this sub dataset because they are more relevant to our daily lives, and the demo would be in a similar setting.
- depth_vis: output inferred pictures

The rest are less relevant. They are either unsuccessful attempts or previous versions of the current working script for fine-tuning, PTQ, QAT, profiler logs, or result logs.

## Important notes
- I ran in to trouble with torch not using cuda because of version mismatch and fixed that by uninstall torch and reinstalling it with specific cuda. I used `torch 2.8.0+cuda128` and `torchao 0.13.0`
- Please refer to the main [DepthAnythingV2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2) and [Hypersim GitHub](https://github.com/apple/ml-hypersim) for more information about the baseline and dataset.
- The tutorial that helped me finish the project is [here](https://docs.pytorch.org/ao/stable/finetuning.html)
- The explanation of quantization concepts that help me finish the poster and report is [here](https://www.youtube.com/watch?v=0VdNflU08yA&t=2434s)
