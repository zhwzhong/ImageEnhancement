# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   option.py
@Time    :   2019/3/1 17:47
@Desc    :
"""
import os
import argparse
# import template

parser = argparse.ArgumentParser(description='Model for ImageRestoration')

parser.add_argument('--template', default='.', help='You Can Set Various Template in option.py')

# Hardware Specifications
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='Use CPU only')
parser.add_argument('--num_GPUs', type=int, default=1, help='Number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='Random Seed')
parser.add_argument('--cuda_name', type=str, default='0')

# Data Specifications
parser.add_argument('--data_name', type=str, default='ImageSuperResolution')
parser.add_argument('--dir_data', type=str, default='/userhome/MyData')    # 数据路径
parser.add_argument('--ext', type=str, default='sep', help='Dataset File Extension')
parser.add_argument('--scale', type=str, default=1, help='Super Resolution Scale')
parser.add_argument('--train_patch_size', type=int, default=32, help='Train Patch Size')
parser.add_argument('--val_patch_size', type=int, default=32, help='Val Patch Size')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--no_augmentation', action='store_true', help='Do not Use Data Augmentation')

# Model Specifications
parser.add_argument('--model', type=str, default='MyNet', help='Model Name')
parser.add_argument('--act', type=str, default='ReLU', help='Activation Function')
parser.add_argument('--num_features', type=int, default=64, help='Number of Feature Maps')
parser.add_argument('--re_scale', type=float, default=1, help='Residual Scaling')
parser.add_argument('--shift_mean', type=bool, default=False, help='Subtract Pixel Mean From the Input')
parser.add_argument('--dilation', type=int, default=1, help='Use Dilation Convolution')
parser.add_argument('--stride', type=int, default=1, help='Stride For Convolution Operation')
parser.add_argument('--norm', type=str, default=None, help='Normalization Method')
parser.add_argument('--kernel_size', type=int, default=3, help='Kernel Size')
parser.add_argument('--high_pass', type=bool, default=False, help='High Pass For Training Neural Network')
parser.add_argument('--trade_off', type=float, default=1, help="Trade Off for Gradient Sensitive Loss")
parser.add_argument('--filter_flow', type=bool, default=False, help='Use Predictive Filter Flow')
parser.add_argument('--in_channels', type=int, default=3, help='Input Channel Number for CNN')
parser.add_argument('--out_channels', type=int, default=3, help='Output Channels Number for CNN')
parser.add_argument('--load_best', type=bool, default=False, help='Load Best Epoch for Test')
parser.add_argument('--load_train_best', type=bool, default=False, help='Load Best Epoch From Train')

# Training Specifications
parser.add_argument('--train_batch_size', type=int, default=64, help='Train Batch Size')
parser.add_argument('--test_batch_size', type=int, default=32, help='Test Batch Size')
parser.add_argument('--re_load', type=bool, default=True, help='Load Model From Previous Model')
parser.add_argument('--test_every', type=int, default=100, help='Do Test Per Every N Step')
parser.add_argument('--epochs', type=int, default=300, help="Number of Epoch For Train")
parser.add_argument('--self_ensemble', type=bool, default=False, help='Use Self-Ensemble Method For Test')
parser.add_argument('--test_only', type=bool, default=False, help='Set This Opinion To Test Model')
parser.add_argument('--save_test_results', default=False, help='Save Test Process Results')

# Optimization Specification
parser.add_argument('--weight_init', type=float, default=0.01, help='Weight Initialize For Model')
parser.add_argument('--init_model', default=False, type=bool, help='USE kaiming Norm for Weight Initialize')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--decay', type=int, default=150, help='Learning Rate Decay Step')
parser.add_argument('--optimizer', type=str, default='ADAM', help='Optimizer To Use For Step Decay')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
parser.add_argument('--gradient_clip', type=float, default=0, help='Gradient Clipping Threshold (0 = no clipping)')

# Loss Specifications
parser.add_argument('--loss', type=str, default='1*L1', help='Loss Function Configuration')  # GradientLoss, TVLoss
parser.add_argument('--save_models', type=bool, default=True, help='Save All Intermediate Models')

# EDSR Specification
parser.add_argument('--num_resblocks', type=int, default=64, help='Number of ResBlocks')

# MyNet Specification
parser.add_argument('--num_groups', type=int, default=4, help='Number of RRDB')
parser.add_argument('--num_blocks', type=int, default=23, help='Number of RDB in Each RRDB')
parser.add_argument('--num_conv', type=int, default=4, help='Number of Convolution Operation in each RDB')
parser.add_argument('--growth_rate', type=int, default=32, help='G')

# Attention Model
parser.add_argument('--add_inception', action='store_true', help='USE Inception As Local Feature Fused Function')
parser.add_argument('--new_inception', type=bool, default=False, help='New Inception With Big Feature and conv 3x3')
parser.add_argument('--no_spatial', action='store_true', help='No Spatial Attention')
parser.add_argument('--attention_block_name', type=str, default='CBAM', help='Attention Block Name For Net')
parser.add_argument('--channel_pool_type', default='avg+max+var')  # lp, var, lse
parser.add_argument('--spatial_attention_type', type=str, default='Conv', help='Spatial Attention For Net[Conv Pool]')

parser.add_argument('--channel_rca', type=bool, default=False, help='Residual Attention As Channel Attention')
parser.add_argument('--rca_leaky', type=bool, default=False)
parser.add_argument('--rca_bn', type=bool, default=False)
parser.add_argument('--spatial_bn', action='store_true')
parser.add_argument('--torch_normalize', action='store_true')

parser.add_argument('--CBAM', action='store_true', help='USE Default CBAM As Attention BLock') # 使用时为True

# EDSR Specification

# Log Specifications

args = parser.parse_args()
# template.set_template(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
# MyNet Specification

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False



