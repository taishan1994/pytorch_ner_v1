import sys
sys.path.append('..')
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
# 这里要显示的引入BertFeature，不然会报错
from preprocess.bertNerNorProcess import BertFeature
from utils import commonUtils


class NerDataset(Dataset):
    def __init__(self, features):
        # self.callback_info = callback_info
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels) for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
        }

        data['labels'] = self.labels[index]

        return data
