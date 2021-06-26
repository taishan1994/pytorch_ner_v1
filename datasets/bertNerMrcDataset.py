import sys

sys.path.append('..')
import torch
from torch.utils.data import Dataset,DataLoader
# 这里要显示的引入BertFeature，不然会报错
from preprocess.bertNerMrcProcess import MRCBertFeature
from utils import commonUtils


class NerMrcDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.start_ids = [torch.tensor(example.start_ids).long() for example in features]
        self.end_ids = [torch.tensor(example.end_ids).long() for example in features]


    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}


        data['start_ids'] = self.start_ids[index]
        data['end_ids'] = self.end_ids[index]

        return data
