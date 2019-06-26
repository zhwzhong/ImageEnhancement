# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   template.py
@Time    :   2019/3/1 17:47
@Desc    :
"""

def set_template(args):
    if args.model_name == 'DnCNN':
        args.kernel_size = 3
        args.features = 64
        args.in_channels = 1
        args.num_of_layers = 17








