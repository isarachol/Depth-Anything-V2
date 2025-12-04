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

from torch.utils.data import DataLoader

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.hypersim import Hypersim
from add_v_cbar import add_v_cbar

# force using cpu
# torch.cuda.is_available = lambda: False

# if __name__ == '__main__':
def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    # parser.add_argument('--img-path', type=str)
    parser.add_argument('--dataset', type=str, default='HyperSim')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./depth_vid/qat')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vits.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    # when finetuned without distribute --> key names remanined the same
    if 'checkpoints' in args.load_from:
        name = '_checkpoint_'
    elif 'finetuned_v1' in args.load_from:
        name = '_finetuned_v1_'
    elif 'finetuned_v2' in args.load_from:
        name = '_finetuned_v2_'
    elif 'quantized_int8_v1' in args.load_from:
        name = '_qat_int8_gs32_'
    elif 'quantized_int8_v2' in args.load_from:
        name = '_qat_int8_gs128_'
    else:
        raise NotImplementedError



    if 'QAT' in args.load_from:
        # set safe globals
        import torchao
        torch.serialization.add_safe_globals([torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor])
        
        new_state_dict = torch.load(args.load_from, map_location='cpu')

        if 'finetuned' in args.load_from: # load right away
            depth_anything.load_state_dict(new_state_dict)
        else:
            from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
            from torchao.quantization.qat import QATConfig

            if '32' in args.load_from: # load finetuned
                base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
                finetuned_path = "QAT/int8_group32/vits_HyperSim_finetuned_v1.pth"
                finetined_state_dict = torch.load(finetuned_path, map_location='cpu')
                depth_anything.load_state_dict(finetined_state_dict)
            elif '128' in args.load_from:
                base_config = Int8DynamicActivationInt4WeightConfig(group_size=128)
                finetuned_path = "QAT/int8_group128/vits_HyperSim_finetuned_v2.pth"
                finetined_state_dict = torch.load(finetuned_path, map_location='cpu')
                depth_anything.load_state_dict(finetined_state_dict)
            else:
                raise NotImplementedError
            
            quantize_(depth_anything, QATConfig(base_config, step="prepare")) # prepare
            quantize_(depth_anything, QATConfig(base_config, step="convert")) # quantize
            depth_anything.load_state_dict(new_state_dict) # now load quantized state
    else:
        new_state_dict = torch.load(args.load_from, map_location='cpu')
        depth_anything.load_state_dict(new_state_dict) # added strict = False
    
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
        testset = Hypersim('dataset/splits/HyperSim/test.txt', 'test')
    else:
        raise NotImplementedError
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=0) # load sequancially, process 1 fig at a time, 
        
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    elapsed_t = 0
    for i, sample in enumerate(testloader):
        print(f'Progress {i+1}/{len(testloader)}:')

        filename, depth_true = sample['image_path'][0], sample['depth'][0]
        depth_true = depth_true.numpy()
        raw_image = cv2.imread(filename)
        
        start_t = time.time()
        depth_inferred = depth_anything.infer_image(raw_image, args.input_size) # metric
        end_t = time.time()
        elapsed_t += end_t - start_t

        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth_inferred)
        
        # Isara: extract info
        vmax_true = depth_true.max()
        vmin_true = depth_true.min()
        vmax = depth_inferred.max()
        vmin = depth_inferred.min()
        height, width = depth_inferred.shape

        # Isara: For adding color bar and edges
        edge_thickness = round(height/20)
        cbar_width = round(width/15)
        white_edge_v = np.ones((edge_thickness, width, 3), dtype=np.uint8) * 255
        raw_image = cv2.vconcat([white_edge_v, raw_image, white_edge_v])

        # make color relative
        depth_inferred = (depth_inferred - depth_inferred.min()) / (depth_inferred.max() - depth_inferred.min()) * 255.0
        depth_inferred = depth_inferred.astype(np.uint8)

        depth_true = (depth_true - depth_true.min()) / (depth_true.max() - depth_true.min()) * 255.0
        depth_true = depth_true.astype(np.uint8)
        
        if args.grayscale:
            depth_inferred = np.repeat(depth_inferred[..., np.newaxis], 3, axis=-1) # repeat 3 layers --> grey scaled
            v_cbar = add_v_cbar(None, vmax, vmin, cbar_width, height, edge_thickness)
        else:
            depth_true = (cmap(depth_true)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            v_cbar_true = add_v_cbar(cmap, vmax_true, vmin_true, cbar_width, height, edge_thickness)
            depth_inferred = (cmap(depth_inferred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            v_cbar = add_v_cbar(cmap, vmax, vmin, cbar_width, height, edge_thickness)
        
        # Isara: add edges and color bar
        depth_inferred = cv2.vconcat([white_edge_v, depth_inferred, white_edge_v])
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        depth_inferred = cv2.hconcat([depth_inferred, split_region, v_cbar])

        depth_true = cv2.vconcat([white_edge_v, depth_true, white_edge_v])
        depth_true = cv2.hconcat([depth_true, split_region, v_cbar_true])

        # Isara: for edges along width direction
        white_edge_h = np.ones((height+2*edge_thickness, edge_thickness, 3), dtype=np.uint8) * 255

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + name + '.png') # + datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        if args.pred_only:
            depth_inferred = cv2.hconcat([white_edge_h, depth_inferred, white_edge_h]) # Isara
            cv2.imwrite(output_path, depth_inferred)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([white_edge_h, raw_image, split_region, depth_true, split_region, depth_inferred, white_edge_h]) # Isara added w_edge
            
            cv2.imwrite(output_path, combined_result)

    print(f'Inferring depth of {len(testloader)} images in {elapsed_t:.2f} sec = {elapsed_t/len(testloader):.4f} s/img')

if __name__ == '__main__':
    main()
