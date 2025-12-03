import argparse
import logging
import warnings
import cv2
import glob
import matplotlib
import numpy as np
import os
import pprint
import torch
import time
import cProfile
import pstats

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from depth_anything_v2.dpt import DepthAnythingV2
from util.metric import eval_depth
from util.utils import init_log
from add_v_cbar import add_v_cbar


def main():
    start_tot_time = time.time()
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation Test CPU Only')
    
    parser.add_argument('--input-size', type=int, default=518)
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
      
    parser.add_argument('--dataset', default='HyperSim', choices=['hypersim', 'vkitti', 'HyperSim'])
    parser.add_argument('--img-size', default=518, type=int)
    parser.add_argument('--min-depth', default=0.001, type=float)
    parser.add_argument('--save-path', type=str, required=True)
    
    args = parser.parse_args()

    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    all_args = {**vars(args), 'ngpus': 0} # world_size
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)
        
    size = (args.img_size, args.img_size)

    if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
        testset = Hypersim('metric_depth/dataset/splits/HyperSim/test.txt', 'test', size=size)
    else:
        raise NotImplementedError
    # testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True) #, sampler=testsampler)
    
    DEVICE = 'cpu' #'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    # extract state dict from pre trained model
    if ('PTQ' in args.load_from):
        model_ep = torch.export.load(args.load_from)
        model = model_ep.module()
    else:
        if ('checkpoints' in args.load_from):
            new_state_dict = torch.load(args.load_from, map_location='cpu')
        # elif ('PTQ' in args.load_from):
        #     quantized_state_dict = torch.load(args.load_from, map_location='cpu')
        #     new_state_dict = {}
        #     wrong_keys = []

        #     for key, val in quantized_state_dict.items():
        #         # if '' in key:
        #         # new_key = 
        #         wrong_keys.append(key)

        #     with open('wrong_keys.txt', 'w') as f:
        #         for key in wrong_keys:
        #             f.write(key + '\n')

        else: # if pretrained (by distributed), rename keys
            pretrained_state_dict = torch.load(args.load_from, map_location='cpu')['model']
            new_state_dict = {}

            for key, val in pretrained_state_dict.items():
                new_key = key.replace("module.", "", 1)
                new_state_dict[new_key] = val

        model.load_state_dict(new_state_dict)

    model = model.to(DEVICE).eval() # set to eval mode
    
    results = {'d1': torch.tensor([0.0]), 'd2': torch.tensor([0.0]), 'd3': torch.tensor([0.0]), 
                'abs_rel': torch.tensor([0.0]), 'sq_rel': torch.tensor([0.0]), 'rmse': torch.tensor([0.0]), 
                'rmse_log': torch.tensor([0.0]), 'log10': torch.tensor([0.0]), 'silog': torch.tensor([0.0]), 
                'time': torch.tensor([0.0])}
    nsamples = torch.tensor([0.0])
    
    for i, sample in enumerate(testloader):
        
        img, depth, valid_mask = sample['image'].float(), sample['depth'][0], sample['valid_mask'][0]
        
        with torch.no_grad():
            start_time = time.time()
            pred = model(img)
            end_time = time.time()
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        
        if valid_mask.sum() < 10:
            continue
        
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask]) # evaluate with metrics
        cur_results['time'] = end_time - start_time
        
        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1
    
    end_tot_time = time.time()
    tot_time = end_tot_time - start_tot_time
    epoch = 1

    logger.info(f'Performance of model from "{args.load_from}"')
    logger.info(f'Measured over "{nsamples.item()}" samples of training data from "{args.dataset}" dataset')

    logger.info(f'Using "{DEVICE}"')
    logger.info(f'Total run time is {tot_time:.3f} s')
    logger.info('Time is in seconds per sample (only for inference step)')
    logger.info(' ')

    logger.info('==================================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
    logger.info('==================================================================================================')
    print()
        
    for name, metric in results.items():
        writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
    
if __name__ == '__main__':
    main()
