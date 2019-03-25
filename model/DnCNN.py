# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   DnCNN.py
@Time    :   2019/1/16 23:24
@Desc    :
"""
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, opt):
        super(DnCNN, self).__init__()
        kernel_size = opt.kernel_size
        padding = (kernel_size - 1) // 2
        features = opt.features
        channels = opt.in_channels
        num_of_layers = opt.num_of_layers
        norm = opt.norm
        layers = [nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                            bias=False), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            if norm == 'BN':
                layers.append(nn.BatchNorm2d(features))
            elif norm == 'IN':
                layers.append(nn.InstanceNorm2d(features))

            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        res = x
        out = self.dncnn(x)
        return out + res
