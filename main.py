#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Author 柳广鹏
import time
import torch
import numpy as np
from importlib import import_module  # 动态加载
import argparse
import utils

parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classsification')
parser.add_argument('--model', type=str, default='Brucebert',
                    help='choose a model BruceBert, BruceBertCNN, BruceBertRNN, BruceBertDPCNN')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集地址
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.bulid_dataset(config)
    # 迭代器
    train_iter = utils.bulid_iterator(train_data, config)
    dev_iter = utils.bulid_iterator(dev_data, config)
    test_iter = utils.bulid_iterator(test_data, config)
