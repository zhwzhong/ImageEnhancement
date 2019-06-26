#!/bin/bash
# MyNet --no_augmentation False --CBAM False --add_inception False --no_spatial False  torch_normalize
# --channel_rca False --re_load False --init_model False  --new_inception   False  --rca_leaky False --rca_bn False --spatial_bn False
# Test  --load_best True  --self_ensemble False --test_only False

python main.py --cuda_name '2, 3, 1' --num_GPUs 3 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 20 --num_conv 4 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --add_inception --no_augmentation \
--lr 1e-4 --decay 10 --loss '1*L1+2*GradientLoss' --no_spatial  \
--attention_block_name 'RAM' --channel_pool_type 'avg+max+var' \
--spatial_attention_type 'Conv' --num_resblocks 32

python main.py --cuda_name '0, 4, 5' --num_GPUs 3 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 20 --num_conv 4 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --add_inception --no_augmentation \
--lr 1e-4 --decay 10 --loss '1*L1+2*GradientLoss' --no_spatial  \
--attention_block_name 'RAM' --channel_pool_type 'avg+max+var' --spatial_bn \
--spatial_attention_type 'Conv' --num_resblocks 32