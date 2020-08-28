#!/usr/bin/python
# -*- coding: UTF-8 -*-

#Author 柳广鹏

import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    参数配置
    """
    def __init__(self, dataset):
        self.model_name = 'BruceBert'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数，轮次
        self.num_epochs = 3
        # batch_size,每个批次的数目
        self.batch_size = 128
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.bert_path = 'bert_pretrain'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐层层个数，要跟预训练模型中的参数一致
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 加载预训练模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 预训练模型的参数梯度不冻结，进行对预训练的微调
        for param in self.bert.parameters():
            param.requires_grad = True
        # 下游分类任务使用线性分类器
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[128,32] [batch_size, pad_size]
        mask = x[2]  # 对padding部分进行mask shape[128,32] [batch_size, pad_size]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True)  # shape [128,768] [
        # batch_size, hidden_size]
        out = self.fc(pooled)  # shape [128,10]
        return out