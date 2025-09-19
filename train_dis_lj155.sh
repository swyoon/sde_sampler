#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=LJ55 solver=dis_no_score_egnn_lj  train_batch_size=8 eval_batch_size=8
done