#!/usr/bin/env bash
python main.py --cuda_name '0' --num_GPUs 1 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 23 --num_conv 4 --growth_rate 32 \
--act 'ReLU'  --re_scale 0.2 --add_inception --no_augmentation \
--lr 1e-4 --decay 10 --loss '1*L1+2*GradientLoss' \
--attention_block_name 'CBAM' --channel_pool_type 'avg+max+var' \
--spatial_attention_type 'Conv' --num_resblocks 32 &

python main.py --cuda_name '1' --num_GPUs 1 --data_name 'ImageRestoration'  --model 'MyNet' \
--num_features 64  --growth_rate 32 --num_groups 3 --num_blocks 23 --num_conv 4 --growth_rate 32 \
--act 'ReLU'  --re_scale 1 --add_inception  --no_augmentation \
--lr 1e-4 --decay 10 --loss '1*L1+2*GradientLoss' \
--attention_block_name 'CBAM' --channel_pool_type 'avg+max+var' \
--spatial_attention_type 'Conv' --num_resblocks 32
