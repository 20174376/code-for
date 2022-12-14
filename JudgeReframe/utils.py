from torch.utils.data import TensorDataset, DataLoader
import csv
import torch
from tqdm import tqdm
import time, os
from datetime import timedelta
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'

def load_dataset(path):
    contents = []
    f = open(path, "r")
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        linelist = line.split("\t")
        if len(linelist) == 2:
            original = linelist[0]
            reframe = linelist[1]
        else:
            print("maybe wrong!")
            original = linelist[0]
            reframe = ""
        if len(linelist) == 3:
            label = int(linelist[2])
        else:
            label = -1
        sent = SEP.join([original, reframe])
        contents.append([sent, label])
    # for line in tqdm(f):
    #     line = line.strip()
    #     linelist = line.split("\t")
    #     original = linelist[0]
    #     reframe = linelist[1]
    #     if len(linelist) == 3:
    #         label = int(linelist[2])
    #     else:
    #         label = -1
    #     sent = SEP.join([original, reframe])
    #     contents.append([sent, label])
    f.close()
    return contents


def build_dataset(config):
    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test

def build_iterator(dataset, config, istrain):
    sent = torch.LongTensor([temp[0] for temp in dataset])
    labels = torch.LongTensor([temp[1] for temp in dataset])
    train_dataset = TensorDataset(sent, labels)
    if istrain:
        train_loader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, 
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.batch_size,
                                 drop_last=True)
    return train_loader

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def gettoken(config, sent):
    # tokenizer = config.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, do_lower_case=True)
    encode_result = tokenizer(list(sent), padding='max_length', truncation=True, max_length=config.max_length)
    input_ids = torch.tensor(encode_result['input_ids'])
    attention_mask = torch.tensor(encode_result['attention_mask'])
    # #用来标明属于哪一个句子
    # type_ids = torch.tensor(encode_result['token_type_ids'])
    
      ### 这个是双句 的
      ### [SEP]在BERT为102，在ERINE为2
    type_ids = []
    for input in input_ids:
        index = 0
        token_type_ids = []
        for i in range(len(input)):
            #[SEP]分隔符
            if input[i] != torch.tensor(102):
                token_type_ids.append(0)
            else:
                token_type_ids.append(0)
                break
        while(len(token_type_ids) < len(input)):
            # print(input[index])
            token_type_ids.append(1)

        # print(len(token_type_ids))
        assert len(token_type_ids) == config.max_length
        type_ids.append(token_type_ids)
    
    type_ids = torch.tensor(type_ids)          
    
    
    position_ids = []
    for j, ids in enumerate(input_ids):
        position_id = list(range(config.max_length))
        position_ids.append(position_id)
    position_ids = torch.tensor(position_ids)
    return input_ids, attention_mask, type_ids, position_ids
