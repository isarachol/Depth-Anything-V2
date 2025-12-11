import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time
import cProfile
import pstats
from datetime import datetime

import torchao
from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
from torchao.quantization.qat import QATConfig

from depth_anything_v2.dpt import DepthAnythingV2
from add_v_cbar import add_v_cbar

torch.serialization.add_safe_globals([torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor])

# force using cpu
# torch.cuda.is_available = lambda: False

parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

parser.add_argument('--img-path', type=str)
parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--outdir', type=str, default='./vis_depth')

parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vits.pth')
parser.add_argument('--max-depth', type=float, default=20)

parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'Load from: {args.load_from}')

def get_model(state_dict_path):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    new_state_dict = torch.load(state_dict_path, map_location='cpu')

    if 'finetuned' in state_dict_path or 'checkpoint' in state_dict_path: # load right away
            depth_anything.load_state_dict(new_state_dict)
            if 'finetuned_v2' in state_dict_path:
                name = 'finetuned2'
            elif 'finetuned_v1' in state_dict_path:
                name = 'finetuned1'
            else:
                name = 'checkpoint'
    else:

        if '32' in state_dict_path: # load finetuned
            base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
            finetuned_path = "QAT/int8_group32/vits_HyperSim_finetuned_v1.pth"
            finetined_state_dict = torch.load(finetuned_path, map_location='cpu')
            depth_anything.load_state_dict(finetined_state_dict)
            name = 'quantized1'
        elif '128' in state_dict_path:
            base_config = Int8DynamicActivationInt4WeightConfig(group_size=128)
            finetuned_path = "QAT/int8_group128/vits_HyperSim_finetuned_v2.pth"
            finetined_state_dict = torch.load(finetuned_path, map_location='cpu')
            depth_anything.load_state_dict(finetined_state_dict)
            name = 'quantized2'
        else:
            raise NotImplementedError
        
        quantize_(depth_anything, QATConfig(base_config, step="prepare")) # prepare
        quantize_(depth_anything, QATConfig(base_config, step="convert")) # quantize

    depth_anything.load_state_dict(new_state_dict)
    
    depth_anything = depth_anything.to(DEVICE).eval()

    return depth_anything, name


def main():
    
    depth_anything, name = get_model(args.load_from)
    print(f'Using model: {name}')
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    elapsed_t = 0
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)
        print(f'image dimensions: {raw_image.shape}')
        
        start_t = time.time()
        depth = depth_anything.infer_image(raw_image, args.input_size) # metric
        end_t = time.time()
        elapsed_t += end_t - start_t

        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)
        
        # Isara: extract info
        vmax = depth.max()
        vmin = depth.min()
        height, width = depth.shape

        # Isara: For adding color bar and edges
        edge_thickness = round(height/20)
        cbar_width = round(width/15)
        white_edge_v = np.ones((edge_thickness, width, 3), dtype=np.uint8) * 255
        raw_image = cv2.vconcat([white_edge_v, raw_image, white_edge_v])

        # make color relative
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1) # repeat 3 layers --> grey scaled
            v_cbar = add_v_cbar(None, vmax, vmin, cbar_width, height, edge_thickness)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            v_cbar = add_v_cbar(cmap, vmax, vmin, cbar_width, height, edge_thickness)
        
        # Isara: add edges and color bar
        depth = cv2.vconcat([white_edge_v, depth, white_edge_v])
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        depth = cv2.hconcat([depth, split_region, v_cbar])
        # Isara: for edges along width direction
        white_edge_h = np.ones((height+2*edge_thickness, edge_thickness, 3), dtype=np.uint8) * 255

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + f'_{name}_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png')
        if args.pred_only:
            depth = cv2.hconcat([white_edge_h, depth, white_edge_h]) # Isara
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([white_edge_h, raw_image, split_region, depth, white_edge_h]) # Isara added w_edge
            
            cv2.imwrite(output_path, combined_result)

    print(f'Inferring depth of {len(filenames)} images in {elapsed_t:.2f} sec = {elapsed_t/len(filenames):.2f} s/img')

if __name__ == '__main__':
    main()
