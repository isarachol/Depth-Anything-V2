#!/bin/bash

python qat.py --bs 4 --epochs 120 2>&1 | tee -a "QAT/$(date +"%Y%m%d_%H%M%S")_QAT.log"
