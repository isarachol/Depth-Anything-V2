#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 15:20:31 2025

@author: isara
"""

# interpret event from summarywriter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

log_dir = "/usr4/cs523aw/isara/depth_estimation/Depth-Anything-V2/metric_depth/exp/HyperSim/ai_001_001/events.out.tfevents.1763682160.scc-202.2201916.0"  # Replace with your actual log directory
event_acc = EventAccumulator(log_dir)

keys = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog'] # , 'time'
x = {}
step = range(120)

event_acc.Reload()

for key in keys:
    scalar_tag = f'eval/{key}'  # Replace with the actual tag name of your scalar
    scalar_events = event_acc.Scalars(scalar_tag)
    
    x[key] = [e.value for e in scalar_events]
    plt.plot(step, x[key])
    
plt.xlabel("Step (0-119)")
plt.ylabel("Value")
plt.legend(keys, loc='upper right', bbox_to_anchor=(1, 0.9))
plt.show()
    
    # for event in scalar_events:
    #     print(f"Step: {event.step}, Value: {event.value}, Wall Time: {event.wall_time}")