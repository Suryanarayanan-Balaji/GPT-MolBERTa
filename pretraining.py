from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import BertTokenizer,  BertConfig, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
import torch
import logging
logging.basicConfig(level = logging.ERROR)
import os
import yaml
from sklearn.model_selection import train_test_split
import numpy as np

location = os.getcwd() # /home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_pretrain.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

if config['model'] == 'bert':
    path = os.path.join(location, 'bert')
    filepath = os.path.join(path, 'pretrain_data_bert.txt')

    vocab_file_directory = os.path.join(path, 'vocab.txt')

    tokenizer = BertTokenizer.from_pretrained(vocab_file_directory)

    configuration = BertConfig(**config['model_bert'])
    
    model = BertForMaskedLM(configuration)

    print(f'Number of parameters: {model.num_parameters()}')

elif config['model'] == 'roberta':
    path = os.path.join(location, 'roberta')
    train_file = os.path.join(path, 'pretrain_data_roberta.txt')

    vocab_file_directory = os.path.join(path, 'tokenizer_folder')

    tokenizer = RobertaTokenizer(
        os.path.join(vocab_file_directory, 'vocab.json'),
        os.path.join(vocab_file_directory, 'merges.txt')
    )

    configuration = RobertaConfig(**config['model_roberta'])
    
    model = RobertaForMaskedLM(configuration)

    print(f'Number of parameters: {model.num_parameters()}')

train_dataset = LineByLineTextDataset( 
    tokenizer = tokenizer,
    file_path = train_file,
    block_size = 128
)

print(f'No. of lines: {len(train_dataset)}')

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)

training_args = TrainingArguments(
    output_dir = path,
    overwrite_output_dir = True,
    num_train_epochs = config['epoch'], # 7
    per_device_train_batch_size = config['batch_size'], # 30
    save_steps = 1000,
    save_total_limit = 5
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator, 
    train_dataset = train_dataset,
    compute_metrics = compute_metrics,
)

trainer.train()
trainer.save_model(path)
