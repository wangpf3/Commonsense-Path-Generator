#!/bin/bash

source ./config/path_generate.config

nohup python -u calc_path_embedding.py \
    --data_dir $data_dir \
    --generator_type $generator_type \
    --batch_size $batch_size \
    --output_len $output_len \
    --context_len $context_len \
    --gpu_device $gpu_device \
    > ./saved_models/debug_save_emb.log 2>&1 &
