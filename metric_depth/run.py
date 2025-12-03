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

from depth_anything_v2.dpt import DepthAnythingV2
from add_v_cbar import add_v_cbar

# force using cpu
torch.cuda.is_available = lambda: False

# if __name__ == '__main__':
def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    # extract state dict from pre trained model
    if 'checkpoints' in args.load_from:
        new_state_dict = torch.load(args.load_from, map_location='cpu')
    else:
        pretrained_state_dict = torch.load(args.load_from, map_location='cpu')['model']
        new_state_dict = {}

        for key, val in pretrained_state_dict.items():
            new_key = key.replace("module.", "", 1)
            new_state_dict[new_key] = val

    depth_anything.load_state_dict(new_state_dict) # added ['model']
    depth_anything = depth_anything.to(DEVICE).eval()
    
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
    
    # start_t = time.time()
    elapsed_t = 0
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
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

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            depth = cv2.hconcat([white_edge_h, depth, white_edge_h]) # Isara
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([white_edge_h, raw_image, split_region, depth, white_edge_h]) # Isara added w_edge
            
            cv2.imwrite(output_path, combined_result)
    end_t = time.time()
    # elapsed_t += end_t - start_t
    print(f'Inferring depth of {len(filenames)} images in {elapsed_t:.2f} sec = {elapsed_t/len(filenames):.2f} s/img')

if __name__ == '__main__':
    cProfile.run('main()', 'profile_results.prof')

    stats = pstats.Stats('profile_results.prof')
    stats.sort_stats('time').print_stats(10)
