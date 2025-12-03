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

from packaging import version

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.hypersim import Hypersim
from util.metric import eval_depth
from util.utils import init_log
from add_v_cbar import add_v_cbar

# force using cpu
force_cpu = True
if force_cpu:
    torch.cuda.is_available = lambda: False

# parse arguments
parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
parser.add_argument('--outdir', type=str, default='./vis_depth')
parser.add_argument('--dataset', default='HyperSim', choices=['hypersim', 'vkitti', 'HyperSim'])
parser.add_argument('--img-size', default=518, type=int)

parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
parser.add_argument('--max-depth', type=float, default=20)
parser.add_argument('--min-depth', default=0.001, type=float)

parser.add_argument('--strict', type=str, default='nonstrict', choices=['strict', 'nonstrict'])
parser.add_argument('--test-lim', type=int, default=None)
parser.add_argument('--pyt-ver', type=str, default='newpyt', choices=['newpyt', 'oldpyt'])

# probaly not use
parser.add_argument('--bs', default=1, type=int)

args = parser.parse_args()

# find device
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: "{DEVICE}"')

# setup logger
logger = init_log('global', logging.INFO)
logger.propagate = 0

all_args = {**vars(args), 'ngpus': 0} # world_size
logger.info('{}\n'.format(pprint.pformat(all_args)))
writer = SummaryWriter(args.outdir, filename_suffix=f'_ptq_{args.pyt_ver}_nonstrict_x86')

# Timer helper function (https://github.com/arikpoz/neural-network-optimization/blob/main/Quantization%20-%20PTQ%20using%20PyTorch%202%20Export%20Quantization%20and%20X86%20Backend.ipynb)
class Timer:
    """
    A simple timer utility for measuring elapsed time in milliseconds.

    Supports both GPU and CPU timing:
    - If CUDA is available, uses torch.cuda.Event for accurate GPU timing.
    - Otherwise, falls back to wall-clock CPU timing via time.time().

    Methods:
        start(): Start the timer.
        stop(): Stop the timer and return the elapsed time in milliseconds.
    """
    
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
            return (time.time() - self.start_time) # s ==> * 1000  # ms

def estimate_latency(model, example_inputs, repetitions=50):
    """
    Returns avg and std inference latency (ms) over given runs.
    """
    
    timer = Timer()
    timings = np.zeros((repetitions, 1))

    # warm-up
    for _ in range(5):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            timer.start()
            _ = model(example_inputs)
            elapsed = timer.stop()
            timings[rep] = elapsed

    return np.mean(timings), np.std(timings)

def print_results(results):
    logger.info('==================================================================================================')
    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v).item() for v in results.values()])))
    logger.info('==================================================================================================')

def print_size_of_model(model, tag=""):
    """
    Prints model size (MB).
    """
    
    torch.save(model.state_dict(), "temp.p")
    size_mb_full = os.path.getsize("temp.p") / 1e6
    print(f"Size ({tag}): {size_mb_full:.2f} MB")
    os.remove("temp.p")

# Testing function
def test(model, testloader, limit=None):
    """
    function for evaluating performance of a model
    """

    results = eval_depth(None, None, True)
    results['time'] = torch.tensor([0.0]) # assume no cuda
    nsamples = 0
    timer = Timer()  
    if limit is None:
        limit = len(testloader)

    for i, sample in enumerate(testloader):
        
        img, depth, valid_mask = sample['image'], sample['depth'][0], sample['valid_mask'][0] #.float()
        
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
    logger.info('Time is in seconds per sample (only for inference step)')
    logger.info(' ')    

    for k in results.keys():
        results[k] = results[k]/nsamples
    
    print_results(results)
    
    return results

def main():
    test_lim = args.test_lim

    # for testing
    if args.strict == 'strict':
        strict=True
    elif args.strict == 'nonstrict':
        strict=False
    else:
        raise NotImplementedError
    
    if args.pyt_ver == 'newpyt':
        newpyt = True
    elif args.pyt_ver == 'oldpyt':
        newpyt = False
    else:
        raise NotImplementedError

    logger.info(f'Using "{DEVICE}"')

    # Set up model
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
        # keys = []

        for key, val in pretrained_state_dict.items():
            new_key = key.replace("module.", "", 1)
            # keys.append(new_key)
            new_state_dict[new_key] = val

        # with open('keys.txt', 'w') as f:
        #     for key in keys:
        #         f.write(key + '\n')

    depth_anything.load_state_dict(new_state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    print_size_of_model(depth_anything, "full") # before quantization

    # set up dataset
    size = (args.img_size, args.img_size) # not really used --> we assume fixed input image size and quantize based on that size

    if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
        trainset = Hypersim('/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/dataset/splits/HyperSim/train.txt', 'calibrate', size=size)
    else:
        raise NotImplementedError
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True) # not meant to train, delete?

    calibration_set = Subset(trainset, range(256)) # pick only the first 256 images for calibration
    calibration_loader = DataLoader(calibration_set, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)

    if args.dataset == 'HyperSim': # Isara: repeat training with subset of HyperSim
        testset = Hypersim('/home/tand/Documents/class/cs523/project/depth_anything_v2/Depth-Anything-V2/metric_depth/dataset/splits/HyperSim/test.txt', 'test', size=size)
    else:
        raise NotImplementedError
    testloader = DataLoader(testset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)
    
    logger.info('Testing original model')
    results = test(depth_anything, testloader, limit=test_lim) # averaged --> quality + time
    
    os.makedirs(args.outdir, exist_ok=True)

    # ===================================================================================
    # POST TRAINING QUANTIZATION 
    # ===================================================================================
    from torch.ao.quantization.quantize_pt2e import (
        prepare_pt2e,
        convert_pt2e,
    )

    import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer

    # batch of 128 images, each with 3 color channels and 32x32 resolution (CIFAR-10)
    example_inputs = (next(iter(trainloader))['image'].to(DEVICE),) #(torch.rand(args.bs, 3, args.img_size, args.img_size).to(DEVICE),) #(next(iter(trainloader))['image'].to(DEVICE),) #(torch.rand(128, 3, 32, 32).to(DEVICE),)
    # example_inputs = (example_inputs,)

    # export the model to a standardized format before quantization
    if newpyt: #version.parse(torch.__version__) >= version.parse("2.5"): # for pytorch 2.5+
        exported_model  = torch.export.export_for_training(depth_anything, example_inputs, strict=strict).module() # failt due to graph breaks (if, .data, unsupported function)
    else: # for pytorch 2.4
        from torch._export import capture_pre_autograd_graph
        exported_model = capture_pre_autograd_graph(depth_anything, example_inputs) 

    # quantization setup for X86 Inductor Quantizer
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    # preparing for PTQ by folding batch-norm into preceding conv2d operators, and inserting observers in appropriate places
    prepared_model = prepare_pt2e(exported_model, quantizer)

    # run inference on calibration data to collect activation stats needed for activation quantization
    def calibrate(model, data_loader):
        torch.ao.quantization.move_exported_model_to_eval(model)
        with torch.no_grad():
            for _, sample in enumerate(data_loader):
                img = sample['image'] #.float()
                model(img.to(DEVICE))
    calibrate(prepared_model, calibration_loader)

    # converts calibrated model to a quantized model
    quantized_model = convert_pt2e(prepared_model)

    # export again to remove unused weights after quantization
    if newpyt: #version.parse(torch.__version__) >= version.parse("2.5"): # for pytorch 2.5+
        quantized_model = torch.export.export_for_training(quantized_model, example_inputs, strict=strict).module()
    else: # for pytorch 2.4
        quantized_model = capture_pre_autograd_graph(quantized_model, example_inputs)
    
    # test quantized
    logger.info('Testing quantized model')
    quantized_results = test(quantized_model, testloader, limit=test_lim)
    print_size_of_model(quantized_model, "quantized")

    # save model
    quantized_export_path = os.path.join(args.outdir, f'{args.encoder}_{args.dataset}_pqt_{args.pyt_ver}_{args.strict}_x86_v1.pth')
    quantized_ep = torch.export.export(quantized_model, example_inputs)
    torch.export.save(quantized_ep, quantized_export_path)

    # ===================================================================================
    # Optimize with CPP
    # ===================================================================================
    # enable the use of the C++ wrapper for TorchInductor which reduces Python overhead
    import torch._inductor.config as config
    config.cpp_wrapper = True

    # compiles quantized model to generate optimized model
    with torch.no_grad():
        optimized_model = torch.compile(quantized_model)

    # evaluate optimized accuracy
    optimized_results = test(optimized_model, testloader, limit=test_lim)

    # save optimized model
    optimized_model_path = os.path.join(args.outdir, f'{args.encoder}_{args.dataset}_pqt_opt_{args.pyt_ver}_{args.strict}_x86_v1.pth')
    optimized_ep = torch.export.export(optimized_model, example_inputs)
    torch.export.save(optimized_ep, optimized_model_path)

    # get optimized model size
    print_size_of_model(optimized_model, "optimized")


if __name__ == '__main__':
    main()
