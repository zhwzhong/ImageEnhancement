# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   EDSR.py
@Time    :   2019/1/22 19:25
@Desc    :
"""
import torch as t
from model.attention_block import AttentionBlock
from model.common import MeanShift, ResBlock, ResBlockA

def make_model(args):
    return EDSR(args)

class EDSR(t.nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
        kernel_size = args.kernel_size
        stride = args.stride
        norm = args.norm
        in_channels = args.in_channels
        num_features = args.num_features
        re_scale = args.re_scale
        dilation = args.dilation
        act = args.act
        self.args = args
        self.sub_mean = MeanShift(rgb_range=1)
        self.add_mean = MeanShift(rgb_range=1, sign=1)

        m_body = [ResBlock(args=args, num_features=num_features, kernel_size=kernel_size, stride=stride, act=act, norm=norm,
                           dilation=dilation, cbam=args.CBAM, re_scale=re_scale) for _ in range(args.num_resblocks)]
        # elif args.res_type == 'WDSRA':
        #     m_body = [ResBlockA(num_features=num_features, expand=args.expand,kernel_size=kernel_size, stride=stride,
        #                         act=act, norm=norm, dilation=dilation, cbam=args.CBAM, re_scale=re_scale)
        #               for _ in range(args.num_resblocks)]

        self.head = t.nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=kernel_size,
                                stride=stride, padding=(kernel_size - 1) // 2, bias=True, dilation=dilation)
        self.tail = t.nn.Conv2d(in_channels=num_features, out_channels=3, kernel_size=kernel_size,
                                stride=stride, padding=(kernel_size - 1) // 2, bias=True, dilation=dilation)

        self.body = t.nn.Sequential(*m_body)

    def forward(self, x):
        res_add = x
        if self.args.shift_mean:
            x = self.sub_mean(x)

        x = self.tail(self.body(self.head(x)))

        if self.args.shift_mean:
            x = self.add_mean(x)
        if self.args.filter_flow:
            return x * res_add
        return x + res_add

