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

# class BertClass(torch.nn.Module):
#     def __init__(self, configuration):
#         super(BertClass, self).__init__()
#         if config['pretrain'] == 'pretraining':
#             self.l1 = BertModel.from_pretrained(path)
#         else:
#             self.l1 = BertModel(config = configuration)
        # self.pre_classifier = torch.nn.Linear(config['model_bert']['hidden_size'], 256)
#         # self.l2 = torch.nn.Linear(768, 512)
#         # self.l3 = torch.nn.Linear(512, 128)
#         # self.l4 = torch.nn.Linear(128, 16)
        # self.norm1 = torch.nn.InstanceNorm1d(256)
#         # self.norm2 = torch.nn.InstanceNorm1d(512)
#         # self.norm3 = torch.nn.InstanceNorm1d(128)

#         # self.dropout = torch.nn.Dropout(0.1)
#         self.classifier = torch.nn.Linear(256, 2)
#         self.regressor = torch.nn.Linear(256, 1)
#         # self.relu = torch.nn.ReLU()


#         # nn.init.xavier_normal_(self.l2.weight)
#         # nn.init.xavier_normal_(self.l3.weight)
#         # nn.init.xavier_normal_(self.l4.weight)
        # nn.init.xavier_normal_(self.pre_classifier.weight)
        # nn.init.xavier_normal_(self.classifier.weight)
        # nn.init.xavier_normal_(self.regressor.weight)


#     def forward(self, input_ids, attention_mask, token_type_ids):
#         output = self.l1(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
#         hidden_state = output[0]
#         pooler = hidden_state[:, 0]
#         pooler = self.pre_classifier(pooler)
#         pooler = self.norm1(pooler)
#         pooler = F.elu(pooler)
#         # pooler = self.l2(pooler)
#         # pooler = self.norm2(pooler)
#         # pooler = F.elu(pooler)
#         # pooler = self.l3(pooler)
#         # pooler = self.norm3(pooler)
#         # pooler = F.elu(pooler)
#         # pooler = self.l4(pooler)
#         # pooler = F.elu(pooler)

#         # pooler = self.dropout(pooler)
#         if config['dataset']['task'] == 'classification':
#             output = self.classifier(pooler)
#         else:
#             output = self.regressor(pooler)
#         return output

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
        output = self.l1(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, output_attentions = True) # Get the shape of this
        # attn = output['attentions'][0]
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        # pooler = self.relu(pooler) # Check if this affects BBBP, BACE, ClinTox
        pooler = self.pre_classifier(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        if config['dataset']['task'] == 'classification':
            output = self.classifier(pooler)
        else:
            output = self.regressor(pooler)
        return output
        # , attn

# model = RobertaClass(configuration)
# # print(model)
# model_path = '/home/suryabalaji/GPT_MolBERTa/finetune/Trial_1/HIV_pretraining_roberta_10:00:55 07-24-2023/model.pth'
# state_dictionary = torch.load(model_path)
# model.load_state_dict(state_dictionary)

# final_attn_weights = model.state_dict()['l1.pooler.dense.weight']
# keys = model.state_dict().keys()
# print(keys)