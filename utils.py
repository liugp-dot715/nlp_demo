from tqdm import tqdm  # 显示进度
import torch
import time
from datetime import timedelta
import pickle as pkl
import os

PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(file_path, config):
    """
        返回结果 4个list ids, lable, ids_len, mask
        :param file_path:
        :param seq_len:
        :return:[(token_ids, int(lable), seq_len, mask)]
    """
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip()
            # 空跳过
            if not line:
                continue
            # 获取标签、语句
            content, lable = line.split('\t')
            # 切词
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            pad_size = config.pad_size
            if pad_size:
                # pad_size是否存在
                if len(token) < pad_size:
                    # 补全mask与置1
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(lable), seq_len, mask))
    return contents


def bulid_dataset(config):
    """
    返回值 train, dev ,test
    :param config:
    :return:
    """
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test


class DatasetIterator(object):
    # to_tensor中得索引可能会变，除非不改变load_dataset得输出结构
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False #记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 批次标记
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device) #样本数据ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device) #标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device) #每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        # 数据集批次划分
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.dataset[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches


def bulid_iterator(dataset, config):
    # device设备
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter