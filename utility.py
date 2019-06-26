# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   utility.py
@Time    :   2019/3/1 17:38
@Desc    :
"""
import math
import os
import re
import shutil
import time

import adabound
import imageio
import numpy as np
import torch.optim as optim
# import matplotlib.pyplot as plt
from skimage.measure import compare_mse, compare_ssim


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class CheckPoint():
    def __init__(self, args):
        self.args = args


def make_optimizer(args, targets):
    # optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdaBound':
        optimizer = adabound.AdaBound(targets.parameters(), lr=args.lr, final_lr=0.1)
    elif args.optimizer == 'AMSGrad':
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'AMSBound':
        optimizer = adabound.AdaBound(targets.parameters(), lr=args.lr, final_lr=0.1, amsbound=True)
    else:
        optimizer = optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print("Learning Rate: {}".format(param_group['lr']))


def create_dir(path):
    if os.path.isdir(path):
        if path.find('result') >= 0:
            shutil.rmtree(path)
        else:
            print('File Already Exist')
            # time.sleep(30)
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def quantize(*img):
    def _quantize(image):
        image = image.detach().cpu().numpy()
        tmp = np.clip((image * 255.).round(), 0, 255)
        return np.transpose(tmp, (1, 2, 0)).astype(np.uint8)

    return [_quantize(a) for a in img]


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_psnr(im_true, im_test, data_range):
    err = compare_mse(im_true, im_test)
    if err == 0:
        return 100
    return 10 * np.log10((data_range ** 2) / err)


def metrics(out, gt, filename, save=False):
    sum_psnr, sum_ssim = 0, 0
    for img_number in range(out.shape[0]):
        tmp_out, tmp_gt = quantize(out[img_number], gt[img_number])
        if save and filename != '':
            imageio.imwrite(str(filename[img_number]), tmp_out)
        sum_psnr += get_psnr(im_true=tmp_gt, im_test=tmp_out, data_range=255)
        sum_ssim += compare_ssim(X=tmp_out, Y=tmp_gt, data_range=255, multichannel=True)
    return sum_psnr / out.shape[0], sum_ssim / out.shape[0]


def get_max_epoch(list_name):
    max_number = 0
    for name in list_name:
        if name.find("best") < 0:
            tmp = int(re.findall(r"\d+", os.path.basename(name))[0])
            if max_number < tmp:
                max_number = tmp
    return max_number


def write_to_tensorboard(args, writer, lr, gt, out, file_name, loss_value, step, start_time, attr):
    psnr, ssim = metrics(out, gt, file_name, args.save_test_results)
    # if attr == 'train' or attr == 'add_image_val':
    #     if attr == 'add_image_val': attr = 'val'
    #     writer.add_image(attr + '_input',
    #                       torchvision.utils.make_grid(lr.detach(), nrow=8, normalize=True, scale_each=True), step)
    #     writer.add_image(attr + '_output',
    #                       torchvision.utils.make_grid(out.detach(), nrow=8, normalize=True, scale_each=True), step)
    #     writer.add_image(attr + '_gt',
    #                       torchvision.utils.make_grid(gt.detach(), nrow=8, normalize=True, scale_each=True), step)
    writer.add_scalar(attr + '_psnr', psnr, step)
    writer.add_scalar(attr + '_ssim', ssim, step)
    writer.add_scalar(attr + '_loss', loss_value.item(), step)
    print('===>Step: {}, {}_loss: {}, {}_psnr: {}, {}_ssim: {}, time_spend: {}'
          .format(step, attr, loss_value.item(), attr, psnr, attr, ssim, time_since(start_time)))


def set_checkpoint_dir(args):
    if args.test_only is False:
        print('Removing Previous Checkpoints and Get New Checkpoints Dir')
        create_dir('./logs/{}'.format(args.file_name))
        create_dir('./checkpoints/{}'.format(args.file_name))
