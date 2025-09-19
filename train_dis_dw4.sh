#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=dw_4 solver=dis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
done