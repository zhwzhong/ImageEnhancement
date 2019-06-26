# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   get_high_pass.py
@Time    :   2019/3/3 19:16
@Desc    :
"""
import os
import cv2
import imageio
import numpy as np
from skimage.measure import compare_psnr, compare_ssim, compare_mse

def get_high_pass(img):
    return img - cv2.boxFilter(img, -1, (9, 9))

import os
from tqdm import tqdm
import shutil
import numpy as np


def copy_img():
    root = '/data/zhwzhong/Data/ImageRestoration/training_data/iphone'
    img_lists = [os.path.join(root, name) for name in os.listdir(root) if name.endswith('jpg')]
    sum_time = 0
    for lr_img in img_lists:
        # img_lists.set_description("Processing %s" % lr_img)
        sum_time += 1
        gt_img = lr_img.replace('iphone', 'canon')
        rand_num = np.random.randint(0, 100)
        dst_gt = '/data/zhwzhong/Data/MyData/ImageRestoration/train_gt/'
        dst_lr = '/data/zhwzhong/Data/MyData/ImageRestoration/train_lr/'
        if rand_num > 90:
            print(sum_time / len(img_lists))
            dst_gt = dst_gt.replace('train', 'val')
            dst_lr = dst_lr.replace('train', 'val')

        shutil.copy(gt_img, dst_gt)
        shutil.copy(lr_img, dst_lr)

if __name__ == '__main__':
    # copy_img()
    dir_root = '/data/zhwzhong/Data/MyData/ImageRestoration'
    print(len(os.listdir(os.path.join(dir_root, 'train_lr'))))
    sum_time = 0
    lr_list = [i for i in os.listdir(dir_root) if i.find('lr') >= 0]
    for img_dir in lr_list:
        if img_dir.find('lr') >= 0:
            print(img_dir)
            res_path = os.path.join(dir_root, img_dir.replace('_lr', '_res_lr'))
            os.makedirs(res_path, exist_ok=True)
            tmp_path = os.path.join(dir_root, img_dir)
            for img_name in os.listdir(tmp_path):
                sum_time += 1
                tmp_img = cv2.imread(os.path.join(tmp_path, img_name))
                res_img = get_high_pass(tmp_img)
                cv2.imwrite(os.path.join(res_path, img_name), res_img)
                if sum_time % 3000 == 0:
                    print("====>Times: {} Image: {} write to : {}".format(sum_time, os.path.join(tmp_path, img_name), os.path.join(res_path, img_name)))
