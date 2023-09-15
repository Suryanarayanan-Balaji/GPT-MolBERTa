import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')
import logging
logging.basicConfig(level = logging.ERROR)
import os

class Data_Processor:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def data_reader (self):
        location = os.getcwd() # /home/suryabalaji/GPT_MolBERTa
        path = os.path.join(location, 'data')
        fil = path + '/' + str(self.dataset) + '_dataset' + '.csv'
        data = pd.read_csv(fil, sep = ',', encoding = 'utf-8')
        data = data.drop('Unnamed: 0', axis = 1) 
        return data

    def canonical (self, dataframe): 
        a, b = [], []
        for i in dataframe['smiles']:
            mol = Chem.MolFromSmiles(i)
            a.append(mol)
            if mol is not None: 
                b.append(Chem.MolToSmiles(mol))
            else:
                b.append('NaN')
        bb = pd.DataFrame(b, columns = ['canonical smiles'])
        df = pd.concat([dataframe, bb], axis = 1)
        df = df[df['canonical smiles'] != 'NaN'].drop(columns = 'canonical smiles', axis = 1).reset_index(drop = True)
        return df
    
    def data_process (self):
        read_data = self.data_reader()
        processed_data = self.canonical(read_data)
        return processed_data

class QueryData(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, col_name, task):
        self.tokenizer = tokenizer
        self.smiles = dataframe['smiles']
        self.text = dataframe['Descriptions']
        self.labels = dataframe[col_name]
        self.max_length = max_length
        self.dtype = torch.long

        if task == "regression":
            self.dtype = torch.float

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
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype = torch.long),
            'mask': torch.tensor(mask, dtype = torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype = torch.long),
            'labels': torch.tensor(self.labels[index], dtype = self.dtype),
            'smiles': self.smiles[index]
        }

def _generate_scaffold(smiles, include_chirality = False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol = mol, includeChirality = include_chirality)
    return scaffold

def generate_scaffolds(dataset, log_every_n = 1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    
    for ind, smiles in enumerate(dataset.smiles):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key = lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

def scaffold_split(dataset, valid_size, test_size, seed = None, log_every_n = 1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

class Data_Split(object):
    
    def __init__(self, 
        dataframe, col_name, tokenizer, max_length, batch_size, num_workers, 
        valid_size, test_size, task, splitting
    ):
        super(object, self).__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.col_name = col_name
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        dataset = QueryData(dataframe = self.dataframe.dropna(subset = [self.col_name]).reset_index(drop = True), tokenizer = self.tokenizer, \
        max_length = self.max_length, col_name = self.col_name, task = self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            dataset, batch_size = self.batch_size, sampler = train_sampler,
            num_workers = self.num_workers, drop_last = False
        )
        valid_loader = DataLoader(
            dataset, batch_size = self.batch_size, sampler = valid_sampler,
            num_workers = self.num_workers, drop_last = False
        )
        test_loader = DataLoader(
            dataset, batch_size = self.batch_size, sampler = test_sampler,
            num_workers = self.num_workers, drop_last = False
        )

        return train_loader, valid_loader, test_loader
