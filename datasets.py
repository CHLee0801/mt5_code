from dis import Instruction
from torch.utils.data import Dataset
import random
import torch
import json
from torch.nn.utils.rnn import pad_sequence
import os
import re
import copy
import pandas as pd

def label_(name, options, answer):
    label = options.index(answer)
    return label

class Pretrain(Dataset):
    def __init__(self, dataset, tokenizer, type_path, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path

        self.whole_dataset = type_path

        self.dataset_name = dataset

        if type_path == 'train':
            self.dataset = pd.read_csv(f'data/{self.dataset_name}/train.csv')
        else:
            self.dataset = pd.read_csv(f'data/{self.dataset_name}/{type_path}.csv')


        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_feature_tokenizer(self, input_, target_, options):
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                            padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")

        data_label = self.whole_dataset 
        return source, targets, data_label, options

    def convert_to_features(self, example_batch, index):
        # prompt evaluation
        label = -1
        options = None
        input_ = example_batch['input']
        target_ = example_batch['output']

        if self.type_path != 'train' and (options != None and self.type_path == 'test'):
            label = options.index(str(target_).strip())

        source, targets, data_label, options = self.convert_to_feature_tokenizer(input_, target_, options)
        return source, targets, data_label, options, label

    def __getitem__(self, index):
        indexed_data = self.dataset.iloc[index]    
        source, targets, data_label, options, label = self.convert_to_features(indexed_data, index)
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        if self.type_path != 'train' and (options != None and self.type_path == 'test'):
            option_list = options
        else:
            option_list = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "data_label": data_label, "option_list": option_list, "label": label}
