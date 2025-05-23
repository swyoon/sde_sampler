#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=cox solver=pis loss.method=kl train_batch_size=512 eval_batch_size=1000
done