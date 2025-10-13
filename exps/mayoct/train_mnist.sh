#!/bin/bash

for NOISE in 0.1; do
    for seeds in 1234; do
        python lpn/train.py \
        --exp_dir exps/mayoct/models_mnist/lpn/s=${NOISE}_noAug_1_${seeds} \
        --dataset_config_path exps/mayoct/configs/mayoct/dataset_mnist.json \
        --model_config_path exps/mnist/configs/mnist/model.json \
        --train_batch_size 200 \
        --dataloader_num_workers 8 \
        --num_steps 158 \
        --num_steps_pretrain 100 \
        --pretrain_lr 1e-3 \
        --lr 1e-4 \
        --num_stages 4 \
        --image_size 28 \
        --num_channels 1 \
        --sigma_noise ${NOISE} \
        --validate_every_n_steps 100 \
        --seed ${seeds}
    done
done