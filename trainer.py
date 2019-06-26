# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   trainer.py
@Time    :   2019/3/1 20:17
@Desc    :
"""
import os
import time
import math
import torch
import imageio
import utility
import numpy as np
from tqdm import tqdm
from decimal import Decimal
import torch.nn.utils as utils
from tensorboardX import SummaryWriter
import torch.nn as nn


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, writer):
        self.args = args
        self.loader_train = loader.loader_train
        self.loader_val = loader.loader_val

        self.loader_test = loader.loader_test
        self.model = my_model
        self.start_time = time.time()
        self.loss = my_loss
        self.writer = writer
        self.step = 0
        self.epoch_number = 0
        self.val_best_psnr = 0
        self.train_best_psnr = 0

        self.optimizer = utility.make_optimizer(args, self.model)
        if args.init_model:
            self.model.apply(self.weights_init_kaiming)
        if args.re_load:
            self.load()

    def train(self):
        self.model.train()
        step = 0
        s
        # self.val()
        for epoch_num in range(self.args.epochs):
            self.epoch_number = epoch_num
            print('Epoch: {}'.format(epoch_num))
            for _, (lr, gt, _) in enumerate(self.loader_train):
                step += 1
                self.optimizer.zero_grad()
                lr, gt = self.prepare(lr, gt)
                output = self.model(lr)
                loss = self.loss(output, gt)
                loss.backward()
                self.optimizer.step()
                if step % self.args.show_every == 0:
                    psnr, ssim = utility.metrics(output, gt, '', False)
                    sum_psnr += psnr
                    sum_ssim += ssim
                    sum_loss += loss.item()
                    sum_img += 1
                    # utility.write_to_tensorboard(self.args, writer=self.writer, lr=lr, gt=gt, out=output, file_name='',
                    #                              loss_value=loss, step=step, start_time=self.start_time, attr='train')
                if step % 1000 == 0:
                    self.step = step
                    self.writer.add_scalar('train_psnr', sum_psnr / sum_img, self.step)
                    self.writer.add_scalar('train_ssim', sum_ssim / sum_img, self.step)
                    self.writer.add_scalar('train_loss', sum_loss / sum_img, self.step)
                    print('===>Step: {}, {}_loss: {}, {}_psnr: {}, {}_ssim: {}, time_spend: {}'
                          .format(step, 'train', sum_loss / sum_img, 'train',  sum_psnr / sum_img, 'train',
                                  sum_ssim / sum_img, utility.time_since(self.start_time)))
                    if self.train_best_psnr < sum_psnr / sum_img:
                        self.train_best_psnr = sum_psnr / sum_img
                        self.save(epoch_num, 'train')
                    sum_loss = sum_psnr = sum_ssim = sum_img = 0
            self.val()
            self.save(epoch_num)
            if (epoch_num + 1) % self.args.decay == 0:
                utility.adjust_learning_rate(self.optimizer, decay_rate=0.3)

    def val(self):
        self.model.eval()
        sum_psnr, sum_ssim, sum_loss, sum_img = 0, 0, 0, 0
        for _ in range(1):     # 重复计算十次取平均
            for _val_step, (lr, gt, file_name) in enumerate(self.loader_val):
                lr, gt = self.prepare(lr, gt)
                output = self.model(lr)
                loss = self.loss(output, gt)
                psnr, ssim = utility.metrics(output, gt, '', False)
                sum_psnr += psnr
                sum_ssim += ssim
                sum_loss += loss.item()
                sum_img += 1
        # writer.add_scalar(attr + '_psnr', psnr, step)
        if sum_psnr / sum_img > self.val_best_psnr:
            self.val_best_psnr = sum_psnr / sum_img
            self.save(self.epoch_number, 'best')

        self.writer.add_scalar('val_psnr', sum_psnr / sum_img, self.step)
        self.writer.add_scalar('val_ssim', sum_ssim / sum_img, self.step)
        self.writer.add_scalar('val_loss', sum_loss / sum_img, self.step)
        print('===>Step: {}, {}_loss: {}, {}_psnr: {}, {}_ssim: {}, time_spend: {}'
              .format(self.step, 'val', sum_loss / sum_img, 'val', sum_psnr / sum_img, 'val',
                      sum_ssim / sum_img, utility.time_since(self.start_time)))

    def test(self):
        self.load()
        self.model.eval()
        print(len(self.loader_test))
        utility.create_dir('./result/{}'.format(self.args.file_name))
        for _, (lr, gt, file_name) in enumerate(self.loader_test):
            lr, _ = self.prepare(lr, gt)
            with torch.no_grad():
            	output = self.model(lr)
            for _num in range(output.shape[0]):
                tmp = utility.quantize(output[_num])
                imageio.imwrite('./result/{}/'.format(self.args.file_name) + str(file_name[_num].numpy()) + '.png', np.array(tmp[0]))
                print('Img: {} saved to ./result/{}/...'.format(str(file_name[_num].numpy()) + '.png', self.args.file_name))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def save(self, epoch_num, last_name=''):
        print('===> Saving models...')
        state = {
            'state': self.model.state_dict(),
            'epoch': epoch_num
        }
        if last_name == 'best':
            torch.save(state, './checkpoints/{}/net_best.pth'.format(self.args.file_name))
        else:
            torch.save(state, './checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(epoch_num)))

    def load(self, epoch_num=0):
        print('===> Loading from checkpoints...')
        if os.path.exists(os.path.join('./checkpoints/{}'.format(self.args.file_name))):

            file_name = os.listdir(os.path.join('./checkpoints/{}'.format(self.args.file_name)))
            if self.args.load_best:
                if self.args.load_train_best:
                    load_name = './checkpoints/{}/net_{}.pth'.format(self.args.file_name, 'train')
                else:
                    load_name = './checkpoints/{}/net_{}.pth'.format(self.args.file_name, 'best')
                if os.path.exists(load_name):
                    checkpoint = torch.load(load_name)
                    print('===> Load best checkpoint data, Epoch: {}'.format(checkpoint['epoch']))
                    self.model.load_state_dict(checkpoint['state'])
                else:
                    print('No Best Model {}'.format(load_name))
            else:
                max_num = utility.get_max_epoch(file_name)
                print(os.path.exists('./checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(max_num))))
                if os.path.exists('./checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(max_num))):
                    checkpoint = torch.load('./checkpoints/{}/net_{}.pth'.format(self.args.file_name, str(max_num)))
                    print('===> Load last checkpoint data, Epoch: {}'.format(checkpoint['epoch']))
                    self.model.load_state_dict(checkpoint['state'])
                else:
                    print('No Max model')
            # self.model.load_state_dict(checkpoint['state'])
        else:
            print('No file To load')

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        a = 0
        nonlinearity = 'relu'
        if self.args.act == 'LeakyReLU':
            a = 0.2
            nonlinearity = 'leaky_relu'
        if classname.find('Conv2d') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=a, mode='fan_in', nonlinearity=nonlinearity) * self.args.weight_init
        # elif classname.find('Linear') != -1:
        #     nn.init.kaiming_normal_(m.weight.data, a=a, mode='fan_in', nonlinearity=nonlinearity) * args.weight_init
        # elif classname.find('BatchNorm') != -1:
        #     # nn.init.uniform(m.weight.data, 1.0, 0.02)
        #     m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        #     nn.init.constant_(m.bias.data, 0.0)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
