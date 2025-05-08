#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for seed in 1 2 3
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=funnel solver=pis loss.method=lv
done