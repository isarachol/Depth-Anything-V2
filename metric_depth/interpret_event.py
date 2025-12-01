#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 15:20:31 2025

@author: isara
"""

# interpret event from summarywriter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# training
model11 = "test_log/ai_001_001/train_vitb/events.out.tfevents.1763682160.scc-202.2201916.0"
model39 = "test_log/ai_003_009/train_vits/events.out.tfevents.1764535757.scc-202.3439401.0"
# testing: probably not for plotting
test39 = "test_log/ai_003_009/test_vits/events.out.tfevents.1764551681.scc-ge2.3394070.0"
test39_vits = "test_log/ai_003_009/test_vits/events.out.tfevents.1764552789.scc-ge2.3396004.0"
test11 = "test_log/ai_001_001/test_vitb/events.out.tfevents.1764548445.scc-202.3465274.0"

log_dir = "/usr4/cs523aw/isara/depth_estimation/Depth-Anything-V2/metric_depth/" + model11  # Replace with your actual log directory
event_acc = EventAccumulator(log_dir)

keys = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog'] # , 'time' # add time for testing
x = {}

event_acc.Reload()

for key in keys:
    scalar_tag = f'eval/{key}'  # Replace with the actual tag name of your scalar
    scalar_events = event_acc.Scalars(scalar_tag)
    
    x[key] = [e.value for e in scalar_events]
    step = range(len(x[key]))
    plt.plot(step, x[key])
    
plt.xlabel("Step (0-119)")
plt.ylabel("Value")
plt.legend(keys, loc='upper right', bbox_to_anchor=(1, 0.9))
plt.show()
    
    # for event in scalar_events:
    #     print(f"Step: {event.step}, Value: {event.value}, Wall Time: {event.wall_time}")