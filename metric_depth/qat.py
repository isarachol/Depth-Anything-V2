import argparse
import logging
import cv2
import glob
import matplotlib
import numpy as np
import os
import sys
import pprint
import torch
import time
import cProfile
import pstats
import random

from packaging import version

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.hypersim import Hypersim
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log
from add_v_cbar import add_v_cbar

# parse arguments
parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

# use default
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--pretrained-from', type=str, default='../checkpoints/depth_anything_v2_vits.pth')
parser.add_argument('--max-depth', type=float, default=20)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--dataset', default='HyperSim', choices=['hypersim', 'vkitti', 'HyperSim'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--save-path', type=str, default='./QAT')
parser.add_argument('--bs', default=1, type=int) # ========================= CHANGE to 128 ====================================
parser.add_argument('--epochs', default=1, type=int) # change to 120
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--test-lim', default=1, type=int)

args = parser.parse_args()

# find device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# setup logger
logger = init_log('global', logging.INFO)
logger.propagate = 0
all_args = {**vars(args), 'ngpus': 0} # world_size
logger.info('{}\n'.format(pprint.pformat(all_args)))
writer = SummaryWriter(args.save_path, filename_suffix=f'_qat')

# Timer helper function (https://github.com/arikpoz/neural-network-optimization/blob/main/Quantization%20-%20PTQ%20using%20PyTorch%202%20Export%20Quantization%20and%20X86%20Backend.ipynb)
class Timer:
    
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda:
            self.starter.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.use_cuda:
            self.ender.record()
            torch.cuda.synchronize()
            return self.starter.elapsed_time(self.ender)  # ms
        else:
            return (time.time() - self.start_time) * 1000 # s ==> * 1000  # ms

def estimate_latency(model, example_input, model_name, repetitions=50):
    """
    Returns avg and std inference latency (ms) over given runs.
    """
    
    timer = Timer()
    timings = np.zeros((repetitions, 1))

    # warm-up
    for _ in range(5):
        _ = model(example_input)

    with torch.no_grad():
        for rep in range(repetitions):
            timer.start()
            _ = model(example_input)
            elapsed = timer.stop()
            timings[rep] = elapsed
    
    print()
    logger.info(f'Estimate latency of {model_name} with same input')
    logger.info(f'Time per sample (ms): {np.mean(timings)} +- {np.std(timings)}')

    return np.mean(timings), np.std(timings)

def print_results(results):
    if len(results) == 10: # with time
        logger.info('==================================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v).item() for v in results.values()])))
        logger.info('==================================================================================================')
    else: # without time
        logger.info('==================================================================================================')
        logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v).item() for v in results.values()])))
        logger.info('==================================================================================================')
    print()

def print_size_of_model(model, tag=""):
    """
    Prints model size (MB).
    """
    
    torch.save(model.state_dict(), "temp.p")
    size_mb_full = os.path.getsize("temp.p") / 1e6
    logger.info(f"Size ({tag}): {size_mb_full:.2f} MB")
    os.remove("temp.p")

def get_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})

    # extract state dict from pre trained model
    # if 'checkpoints' in args.pretrained_from:
    new_state_dict = torch.load(args.pretrained_from, map_location='cpu') # should be fine now
    # else: # for finetuned version
    #     pretrained_state_dict = torch.load(args.load_from, map_location='cpu')
    #     new_state_dict = {}

    #     for key, val in pretrained_state_dict.items():
    #         new_key = key.replace("module.", "", 1)
    #         new_state_dict[new_key] = val

    depth_anything.load_state_dict(new_state_dict)
    return depth_anything.to(DEVICE)

# Training
def train_loop(model, trainloader, valloader):

    criterion = SiLogLoss().to(DEVICE)
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
        logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                    'log10: {:.3f}, silog: {:.3f}'.format(
                        epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                        previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        # trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].to(DEVICE), sample['depth'].to(DEVICE), sample['valid_mask'].to(DEVICE)
            
            if random.random() < 0.5: # randomly flip? 50% & flipped and 50% not
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
            
            pred = model(img) # similar to model.forward(img) BUT with forward hooks and toerh stuff
            
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            
            writer.add_scalar('train/loss', loss.item(), iters)
            
            if i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        # validate
        model.eval()
        logger.info("Validate")
        results = {'d1': torch.tensor([0.0]).to(DEVICE), 'd2': torch.tensor([0.0]).to(DEVICE), 'd3': torch.tensor([0.0]).to(DEVICE), 
                   'abs_rel': torch.tensor([0.0]).to(DEVICE), 'sq_rel': torch.tensor([0.0]).to(DEVICE), 'rmse': torch.tensor([0.0]).to(DEVICE), 
                   'rmse_log': torch.tensor([0.0]).to(DEVICE), 'log10': torch.tensor([0.0]).to(DEVICE), 'silog': torch.tensor([0.0]).to(DEVICE)}
        nsamples = torch.tensor([0.0]).to(DEVICE)
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].to(DEVICE).float(), sample['depth'].to(DEVICE)[0], sample['valid_mask'].to(DEVICE)[0]
            
            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask]) # evaluate with metrics
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
        
        # for k in results.keys():
        #     dist.reduce(results[k], dst=0)
        # dist.reduce(nsamples, dst=0)
        
        print_results(results)
            
        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, f'{args.encoder}_{args.dataset}_train_v1.pth'))

# Testing function
def test(model, testloader, model_name, limit=None):
    """
    function for evaluating performance of a model
    """

    logger.info(f'Testing {model_name} model')
    results = eval_depth(None, None, True)
    results['time'] = torch.tensor([0.0]).to(DEVICE) # assume no cuda
    nsamples = 0
    timer = Timer()  
    if limit is None:
        limit = len(testloader)

    for i, sample in enumerate(testloader):
        
        img, depth, valid_mask = sample['image'].to(DEVICE).float(), sample['depth'][0].to(DEVICE), sample['valid_mask'][0].to(DEVICE) #.float()

        with torch.no_grad():
            timer.start()
            pred = model(img)
            elapsed_time = timer.stop()
            pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
        
        valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
        
        if valid_mask.sum() < 10:
            continue
        
        cur_results = eval_depth(pred[valid_mask], depth[valid_mask]) # evaluate with metrics
        cur_results['time'] = elapsed_time
        
        for k in results.keys():
            results[k] += cur_results[k]
        nsamples += 1

        for name, metric in results.items():
            writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), i)

        if nsamples >= limit:
            break
    
    print()
    logger.info(f'Measured over "{nsamples}" samples of training data from "{args.dataset}" dataset')
    logger.info('Time is in ms per sample (only for inference step)')
    logger.info(' ')    

    for k in results.keys():
        results[k] = results[k]/nsamples
    
    print_results(results)
    
    return results

def main():
    need_train = True
    logger.info(f'Using "{DEVICE}"')

    if need_train:
        test_lim = args.test_lim # =========================================== DELETE ===========================================

        # ===================================================================================
        # Set up dataset
        # ===================================================================================
        size = (args.img_size, args.img_size) # not really used --> we assume fixed input image size and quantize based on that size

        if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
            trainset = Hypersim('dataset/splits/HyperSim/train.txt', 'train', size=size)
        else:
            raise NotImplementedError
        trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True) # not meant to train, delete?

        # calibration_set = Subset(trainset, range(128)) # pick only the first 256 images for calibration
        # calibration_loader = DataLoader(calibration_set, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)

        if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
            valset = Hypersim('dataset/splits/HyperSim/val.txt', 'val', size=size)
        else:
            raise NotImplementedError
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)

        if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
            testset = Hypersim('dataset/splits/HyperSim/test.txt', 'test', size=size)
        else:
            raise NotImplementedError
        testloader = DataLoader(testset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)
        
        os.makedirs(args.save_path, exist_ok=True)
        # Set up model
        checkpoint = get_model()
        depth_anything = checkpoint
        print_size_of_model(depth_anything, "checkpoint") # before quantization
        _ = test(depth_anything, testloader, "rel_checkpoint", limit=test_lim) # averaged --> quality + time

        # ===================================================================================
        # QUANTIZATION AWARE TRAINING
        # ===================================================================================
        from torchao.quantization import quantize_, Int8DynamicActivationInt4WeightConfig
        from torchao.quantization.qat import QATConfig

        # prepare: swap `torch.nn.Linear` -> `FakeQuantizedLinear`
        base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
        quantize_(depth_anything, QATConfig(base_config, step="prepare"))

        # fine-tune --> fake quantization
        train_loop(depth_anything, trainloader, valloader)
        finetuned = depth_anything
        torch.save(finetuned.state_dict(), os.path.join(args.save_path, f'{args.encoder}_{args.dataset}_finetuned_v1.pth'))
        print_size_of_model(depth_anything, "finetuned")
        _ = test(finetuned, testloader, "metric_finetuned", limit=test_lim)

        # convert: swap `FakeQuantizedLinear` -> `torch.nn.Linear`, then quantize using `base_config`
        quantize_(depth_anything, QATConfig(base_config, step="convert"))
        torch.save(depth_anything.state_dict(), os.path.join(args.save_path, f'{args.encoder}_{args.dataset}_quantized_int8_v1.pth'))

        print_size_of_model(depth_anything, "quantized")
        _ = test(depth_anything, testloader, "quantized", limit=test_lim) # averaged --> quality + time
        writer.close()

    else:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
    
        checkpoint = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': args.max_depth})
        finetuned = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': args.max_depth})
        depth_anything = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': args.max_depth})

        metric_cp_path = 'QAT/'
        finetuned_path = ''
        quantized_path = ''

        metric_cp_state = torch.load(metric_cp_path, map_location='cpu')
        finetuned_state = torch.load(finetuned_path, map_location='cpu')
        quantized_state = torch.load(quantized_path, map_location='cpu')

        checkpoint.load_state_dict(metric_cp_state).to(DEVICE)
        finetuned.load_state_dict(finetuned_state).to(DEVICE)
        depth_anything.load_state_dict(quantized_state).to(DEVICE)

    # ===================================================================================
    # COMPARE INFERENCE TIME
    # ===================================================================================

    example_input = torch.rand(1, 3, 518, 686).to(DEVICE).float()

    estimate_latency(checkpoint, example_input, "checkpoint", repetitions=50)
    estimate_latency(finetuned, example_input, "finetuned", repetitions=50)
    estimate_latency(depth_anything, example_input, "quantized", repetitions=50)
    logger.info("End of program")


if __name__ == '__main__':
    main()
