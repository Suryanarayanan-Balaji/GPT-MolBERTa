import torch
from transformers import BertModel, RobertaModel, BertConfig, RobertaConfig
import logging
logging.basicConfig(level = logging.ERROR)
import yaml
import os
import torch.nn.functional as F
import torch.nn as nn

location = os.getcwd() # /home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

if config['model'] == 'bert': 
    path = os.path.join(location, 'bert')
    configuration = BertConfig(**config['model_bert'])

elif config['model'] == 'roberta':
    path = os.path.join(location, 'roberta')
    configuration = RobertaConfig(**config['model_roberta'])

class BertClass(torch.nn.Module):
    def __init__(self, configuration):
        super(BertClass, self).__init__()
        if config['pretrain'] == 'pretraining':
            self.l1 = BertModel.from_pretrained(path)
        else:
            self.l1 = BertModel(config = configuration)
        self.pre_classifier = torch.nn.Linear(config['model_bert']['hidden_size'], 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 2)
        self.regressor = torch.nn.Linear(768, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.l1(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)

        if config['dataset']['task'] == 'classification':
            output = self.classifier(pooler)
        else:
            output = self.regressor(pooler)
        return output

class RobertaClass(torch.nn.Module):
    def __init__(self, configuration):
        super(RobertaClass, self).__init__()
        if config['pretrain'] == 'pretraining':
            self.l1 = RobertaModel.from_pretrained(path)
        else:
            self.l1 = RobertaModel(config = configuration)
        self.pre_classifier = torch.nn.Linear(config['model_roberta']['hidden_size'], 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 2)
        self.regressor = torch.nn.Linear(768, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.l1(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, output_attentions = True)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        if config['dataset']['task'] == 'classification':
            output = self.classifier(pooler)
        else:
            output = self.regressor(pooler)
        return output