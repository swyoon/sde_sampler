#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=dw_4 solver=pis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
done



CUDA_VISIBLE_DEVICES=2 python scripts/main.py seed=0 target=dw_4 solver=pis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
CUDA_VISIBLE_DEVICES=3 python scripts/main.py seed=1 target=dw_4 solver=pis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
CUDA_VISIBLE_DEVICES=4 python scripts/main.py seed=2 target=dw_4 solver=pis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000

CUDA_VISIBLE_DEVICES=4 python scripts/main.py seed=2 target=dw_4 solver=pis_score_egnn_dw4  train_batch_size=128 eval_batch_size=1000