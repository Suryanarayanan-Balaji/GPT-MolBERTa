from transformers import BertTokenizer,  BertConfig, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
import torch
import logging
logging.basicConfig(level = logging.ERROR)
import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset.data_processing import Data_Processor

location = os.getcwd() # /home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_pretrain.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

datasets = ['BBBP', 'Tox21', 'ToxCast', 'SIDER', 'ClinTox', 'BACE', 'HIV', 'MUV', 'ESOL', 'FreeSolv', 'Lipophilicity', 'QM7']
# , 'QM8' , 'QM9']
descriptions = [Data_Processor(elem).data_process()['Descriptions'] for elem in datasets]
pretrain_tokenizer = [item for sublist in descriptions for item in sublist]
pretrain = pd.concat(descriptions, axis = 0, ignore_index = True).to_frame()

if config['model'] == 'bert': # pretrain this as well
    path = os.path.join(location, 'bert')

    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, 'pretrain_data_bert.txt')

    with open(filepath, 'w') as f:
        for sent in pretrain:
            f.write('%s\n' % sent)

    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(
        files = [filepath],
        vocab_size = 50000,
        min_frequency = 2,
        limit_alphabet = 1000
    )
    tokenizer.save_model(path)

    vocab_file_directory = os.path.join(path, 'vocab.txt')

    tokenizer = BertTokenizer.from_pretrained(vocab_file_directory)

    configuration = BertConfig(**config['model_bert'])
    
    model = BertForMaskedLM(configuration)

    print(f'Number of parameters: {model.num_parameters()}')

elif config['model'] == 'roberta':
    path = os.path.join(location, 'roberta')

    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, 'pretrain_data_roberta.txt')

    with open(filepath, 'w') as f:
        for sent in pretrain_tokenizer:
            f.write('%s\n' % sent)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files = [filepath],
        vocab_size = 50000,
        min_frequency = 2, ## Figure this out
        show_progress = True,
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] # mask index = 4
    )

    tokenizer.post_processor = RobertaProcessing(
        sep = ('</s>', tokenizer.token_to_id('</s>')), 
        cls = ('<s>', tokenizer.token_to_id('<s>'))
    )

    tokenizer_path = os.path.join(path, 'tokenizer_folder')
    
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    tokenizer.save_model(tokenizer_path)

    vocab_file_directory = os.path.join(path, 'tokenizer_folder')

    # tokenizer = RobertaTokenizer.from_pretrained(vocab_file_directory)
    tokenizer = RobertaTokenizer(
        os.path.join(vocab_file_directory, 'vocab.json'),
        os.path.join(vocab_file_directory, 'merges.txt')
    )

    configuration = RobertaConfig(**config['model_roberta'])
    
    model = RobertaForMaskedLM(configuration)

    print(f'Number of parameters: {model.num_parameters()}')

class Pretrain_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.text = df['Descriptions']
        self.max_length = max_length
        self.dtype = torch.long

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_length,
            pad_to_max_length = True,
            return_token_type_ids = True
        )
        ids = inputs['input_ids']
        pad = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        non_zero = np.count_nonzero(np.array(ids) != 1) 
        mask_prob = 0.15
        masked_tokens = round(mask_prob * non_zero)
        rand_indices = np.random.choice(np.arange(1, non_zero), masked_tokens, replace = False)

        masked_pad = np.array(pad, dtype = np.int64)
        masked_pad[rand_indices] = 0

        padding = np.arange(non_zero, len(ids))
        labels = np.array(ids, dtype = np.int64)[rand_indices]
        label_array = (np.zeros(len(ids), dtype = np.int64))
        label_array[padding] = -100
        label_array[rand_indices] = labels

        masked_ids = np.array(ids, dtype = np.int64)
        masked_ids[rand_indices] = self.tokenizer.mask_token_id

        return {
            'ids': torch.from_numpy(masked_ids),
            'pad': torch.from_numpy(masked_pad),
            'labels': torch.from_numpy(label_array),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype = torch.long),
        }

validation_proportion = 0.05
num_validation = int(validation_proportion * len(pretrain))
valid_ind = np.random.choice(np.arange(len(pretrain)), num_validation, replace = False)
train_ind = np.array([x for x in range(len(pretrain)) if x not in valid_ind])

train_split = pretrain.loc[train_ind].reset_index(drop = True)
valid_split = pretrain.loc[valid_ind].reset_index(drop = True)

train_dataset = Pretrain_Dataset(train_split, tokenizer, config['max_length'])
valid_dataset = Pretrain_Dataset(valid_split, tokenizer, config['max_length'])

train_dataloader = DataLoader(train_dataset, batch_size = config['train_batch_size'], shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = config['valid_batch_size'], shuffle = False)

optimizer = AdamW(model.parameters(), lr = float(config['learning_rate']), eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * config['pretrain_epochs'])

def gpu_device ():
    if torch.cuda.is_available and config['device'] != 'cpu':
        device = config['device']
    else:
        device = 'cpu'
    print(f'Running on {device}')
    return device
device = gpu_device()

model.to(device)
for epoch in range(config['pretrain_epochs']):
    model.train()
    total_loss = 0.0
    num_train_steps = 0
    
    torch.cuda.empty_cache()
    for ep, batch in tqdm(enumerate(train_dataloader, 0)):
        torch.cuda.empty_cache()
        input_ids = batch['ids'].to(device)
        attention_mask = batch['pad'].to(device)
        labels = batch['labels'].to(device)

        output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_attentions = True)

        loss = output[0]

        num_train_steps += 1
        if ep%500 == 0:
            print(f"Loss: {total_loss/num_train_steps}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{config['pretrain_epochs']}, Loss: {total_loss / len(train_dataloader)}")
    scheduler.step()

model.save_pretrained(path)

model.eval()
valid_loss = 0.0
with torch.no_grad():
    for ep, batch in tqdm(enumerate(valid_dataloader, 0)):
        input_ids = batch['ids'].to(device)
        attention_mask = batch['pad'].to(device)
        labels = batch['labels'].to(device)

        output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_attentions = True)

        loss = output[0]
        valid_loss += loss.item()

    avg_val_loss = valid_loss / len(valid_dataloader)
    print(f"Validation Loss: {avg_val_loss}")