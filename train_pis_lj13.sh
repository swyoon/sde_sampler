#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=LJ13 solver=pis_no_score_egnn  train_batch_size=8 eval_batch_size=8
done