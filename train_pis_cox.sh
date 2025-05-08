#!/bin/bash

export CUDA_VSIBLE_DEVICES=0

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=cox solver=pis loss.method=kl
done