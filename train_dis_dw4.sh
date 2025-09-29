#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

for seed in 1
do
    echo "Running with seed=$seed"
    python scripts/main.py seed=$seed target=dw_4 solver=dis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
done

CUDA_VISIBLE_DEVICES=5 python scripts/main.py seed=0 target=dw_4 solver=dis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
CUDA_VISIBLE_DEVICES=6 python scripts/main.py seed=1 target=dw_4 solver=dis_no_score_egnn_dw4  train_batch_size=64 eval_batch_size=1000
CUDA_VISIBLE_DEVICES=7 python scripts/main.py seed=2 target=dw_4 solver=dis_no_score_egnn_dw4  train_batch_size=128 eval_batch_size=1000


CUDA_VISIBLE_DEVICES=7 python scripts/main.py seed=2 target=dw_4 solver=basic_dis  train_batch_size=128 eval_batch_size=1000


CUDA_VISIBLE_DEVICES=6 python scripts/main.py seed=2 target=dw_4 solver=dds_euler_egnn_dw4 train_batch_size=128 eval_batch_size=1000

CUDA_VISIBLE_DEVICES=7 python scripts/main.py seed=2 target=dw_4 solver=dis_score_egnn_dw4  train_batch_size=128 eval_batch_size=1000