#!/bin/bash


for seed in 1
do
    echo "Running with seed=$seed"
    CUDA_VISIBLE_DEVICES=$seed \
    WANDB_API_KEY=ae46c956323c05750152afe70738312d3a8a2ccf \
    python scripts/main.py seed=$seed target=nice solver=pis loss.method=lv train_batch_size=1024 eval_batch_size=1000 \
    train_steps=30000
done
