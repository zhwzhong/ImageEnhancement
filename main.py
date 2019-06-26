# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   main.py
@Time    :   2019/3/1 19:55
@Desc    :
"""
import loss
import data
import model
import torch
import utility
from option import args
from trainer import Trainer
from tensorboardX import SummaryWriter
torch.manual_seed(args.seed)

# args.test_only = True

utility.set_checkpoint_dir(args)
loader = data.Data(args)
writer = SummaryWriter('./logs/{}'.format(args.file_name))
model = model.Model(args, writer)
print(args)
loss = loss.Loss(args)
train_process = Trainer(args, loader=loader, my_model=model, my_loss=loss, writer=writer)
if args.test_only:
    train_process.test()
else:
    train_process.train()
