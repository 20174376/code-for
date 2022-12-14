from re import A
from turtle import forward
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

class VanillaBert(nn.Module):
    def __init__(self, args):
        super(VanillaBert, self).__init__()
        self.num_labels = 1
        bert_config = AutoConfig.from_pretrained(args.model_path)
        bert_config.return_dict = False
        bert_config.output_hidden_states = True 
        self.bert = AutoModel.from_pretrained(args.model_path, config=bert_config)
        self.dropout = nn.Dropout(args.dropout)
        self.liner = nn.Linear(args.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        last_hidden_state, pooler_output, hidden_output_a = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids, position_ids=position_ids)
        x = self.liner(pooler_output)
        x = torch.sigmoid(x).squeeze(-1)
        return x


class VanillaRoBert(nn.Module):
    def __init__(self, args):
        super(VanillaRoBert, self).__init__()
        self.num_labels = 1
        bert_config = AutoConfig.from_pretrained(args.model_path)
        bert_config.return_dict = False
        bert_config.output_hidden_states = True 
        self.bert = AutoModel.from_pretrained(args.model_path, config=bert_config)
        self.dropout = nn.Dropout(args.dropout)
        self.liner = nn.Linear(args.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        last_hidden_state, pooler_output, hidden_output_a = self.bert(
            input_ids=input_ids, attention_mask=attention_mask,  position_ids=position_ids)
        x = self.liner(pooler_output)
        x = torch.sigmoid(x).squeeze(-1)
        return x






class Bert_Last3andPool(nn.Module):
    def __init__(self, args):
        super(Bert_Last3andPool, self).__init__()
        self.num_labels = 1
        bert_config = AutoConfig.from_pretrained(args.model_path)
        # print(bert_config)
        bert_config.return_dict = False
        bert_config.output_hidden_states = True 
        self.bert = AutoModel.from_pretrained(args.model_path, config=bert_config)

        self.dropout = nn.Dropout(args.dropout)
        self.liner = nn.Linear(args.hidden_size*4, 1)
    
    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        last_hidden_state, pooler_output, hidden_output_a = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids, position_ids=position_ids)
        last_cat = torch.cat(
            (pooler_output, hidden_output_a[-1][:, 0], hidden_output_a[-2][:, 0], hidden_output_a[-3][:, 0]),
            1,
        ) 
        x = self.liner(last_cat)
        x = torch.sigmoid(x).squeeze(-1)
        return x