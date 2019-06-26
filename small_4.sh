#!/bin/bash
# MyNet --no_augmentation False --CBAM False --add_inception False --no_spatial False
# --channel_rca False --re_load False --init_model False
# Test  --load_best True  --self_ensemble False --test_only False

python main.py --cuda_name '0' --num_GPUs 1 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 10 --num_conv 3 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --weight_init 0.01  \
--lr 1e-4 --decay 30 --loss 1*L1 --weight_decay 0 \
--attention_block_name 'CRAM' --add_inception  --channel_pool_type 'avg+max' \
--spatial_attention_type 'Conv' --num_resblocks 32  &

python main.py --cuda_name '1' --num_GPUs 1 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 10 --num_conv 3 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --weight_init 0.01  \
--lr 1e-4 --decay 30 --loss 1*L1+1*GradientLoss --weight_decay 0 \
--attention_block_name 'CRAM' --add_inception  --channel_pool_type 'avg+max' \
--spatial_attention_type 'Conv' --num_resblocks 32  &

python main.py --cuda_name '2' --num_GPUs 1 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 10 --num_conv 3 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --weight_init 0.01  \
--lr 1e-4 --decay 30 --loss 1*L1+1.5*GradientLoss --weight_decay 0 \
--attention_block_name 'CRAM' --add_inception  --channel_pool_type 'avg+max' \
--spatial_attention_type 'Conv' --num_resblocks 32  &

python main.py --cuda_name '3' --num_GPUs 1 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 10 --num_conv 3 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --weight_init 0.01  \
--lr 1e-4 --decay 30 --loss 1*L1+2*GradientLoss --weight_decay 0 \
--attention_block_name 'CRAM' --add_inception  --channel_pool_type 'avg+max' \
--spatial_attention_type 'Conv' --num_resblocks 32  &


