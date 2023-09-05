import pandas as pd
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
import logging
logging.basicConfig(level = logging.ERROR)
import os
import yaml
from dataset.data_processing import Data_Processor
import random 

location = os.getcwd() # /home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_pretrain.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

datasets = ['BBBP' , 'Tox21', 'ToxCast', 'SIDER', 'ClinTox', 'BACE', 'HIV', 'MUV', 'ESOL', 'FreeSolv', 'Lipophilicity', 'QM7' , 'QM8' , 'QM9']

pre = [Data_Processor(elem).data_reader()['Descriptions'].tolist() for elem in datasets]
pretrain_tokenizer = [item for sublist in pre for item in sublist]

if config['model'] == 'bert':
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

elif config['model'] == 'roberta':
    path = os.path.join(location, 'roberta')

    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, 'pretrain_data_roberta.txt')

    with open(filepath, 'w') as f:
        for sent in pretrain:
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
