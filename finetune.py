import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import logging
logging.basicConfig(level = logging.ERROR)

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import yaml
from datetime import datetime
import os

from data_process.data_processing import Data_Split, Data_Processor 
from model.model import BertClass, RobertaClass

class Normalizer(object):
    def __init__(self, col):
        self.mean = col.mean()
        self.stdev = col.std()
    
    def norm (self, col):
        normalized = (col - self.mean)/self.stdev
        return normalized
    
    def denorm (self, norm_col):
        denorm = norm_col*self.stdev + self.mean
        return denorm

class FineTune(object):
    def __init__(self, dataset, config, tokenizer_path):
        self.dataset = dataset
        self.config = config
        self.device = self.gpu_device()
        self.dtype = torch.long

        if self.config['dataset']['task'] == 'regression':
            self.dtype = torch.float

        if self.config['model'] == 'bert':
            self.tokenizer = BertTokenizer(tokenizer_path)

        elif self.config['model'] == 'roberta':
            self.tokenizer = RobertaTokenizer(
                os.path.join(tokenizer_path, 'vocab.json'),
                os.path.join(tokenizer_path, 'merges.txt')
            )
        
        time_ = datetime.now().strftime('%b%d_%H-%M-%S')
        path = config['dataframe'] + '_' + config['pretrain'] + '_' + config['model'] + '_' + time_  
        self.log_dir = os.path.join('finetune', path)
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def gpu_device (self):
        if torch.cuda.is_available and self.config['device'] != 'cpu':
            device = self.config['device']
        else:
            device = 'cpu'
        print(f'Running on {device}')
        return device
    
    def loss_function (self, outputs, labels):
        if self.config['dataset']['task'] == 'classification':
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            return loss
        elif self.config['dataset']['task'] == 'regression':
            if self.config['dataframe'] in ['QM7', 'QM8', 'QM9']:
                criterion = torch.nn.L1Loss()
                loss = criterion(outputs, labels)
                return loss
            else:
                criterion = torch.nn.MSELoss()
                loss = criterion(outputs, labels)
                return loss
    
    def train(self, model, loader, optimizer): 
        train_loss = 0
        num_train_steps = 0

        model.train()

        label_array_batch = []
        prob_array_batch = []

        for ep, data in tqdm(enumerate(loader, 0)):
            ids = data['ids'].to(self.device, dtype = torch.long)
            mask = data['mask'].to(self.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
            labels = data['labels'].to(self.device, dtype = self.dtype)

            outputs = model(ids, mask, token_type_ids) 

            if self.config['dataset']['task'] == 'regression':
                labels = labels.reshape(-1, 1)

            loss = self.loss_function(outputs, labels)
            train_loss += loss.item()

            if self.config['dataset']['task'] == 'classification': 
                probs = torch.softmax(outputs, dim = 1)[:,1]

            elif self.config['dataframe'] in ['FreeSolv', 'ESOL', 'Lipophilicity']: 
                probs = self.normalizer.denorm(outputs)
                labels = self.normalizer.denorm(labels)
            else:
                probs = outputs

            if self.config['device'] == 'cpu':
                label_array_batch.append(labels.numpy())
                prob_array_batch.append(probs.detach().numpy())
            else:
                label_array_batch.append(labels.cpu().numpy())
                prob_array_batch.append(probs.detach().cpu().numpy())

            num_train_steps += 1

            if ep%500 == 0:
                loss_step = (train_loss/num_train_steps)
                print(f'Train Loss per 500 steps: {loss_step}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        label_array_new = np.concatenate(label_array_batch, axis = 0)
        prob_array_new = np.concatenate(prob_array_batch, axis = 0)

        epoch_loss = train_loss/num_train_steps
        print(f"Training Loss Epoch = {epoch_loss}")

        if self.config['dataset']['task'] == 'classification':

            epoch_auc = roc_auc_score(label_array_new, prob_array_new)
            print(f"Training AUC Epoch = {epoch_auc}")

            return epoch_loss, epoch_auc

        elif self.config['dataset']['task'] == 'regression':
            if self.config['dataframe'] in ['QM7', 'QM8', 'QM9']:
                epoch_MAE = mean_absolute_error(label_array_new, prob_array_new)
                print(f"Training MAE Epoch = {epoch_MAE}")

                return epoch_loss, epoch_MAE

            else:
                epoch_RMSE = mean_squared_error(label_array_new, prob_array_new, squared = False)
                print(f"Training RMSE Epoch = {epoch_RMSE}")

                return epoch_loss, epoch_RMSE
            
    def validate(self, model, loader):
        model.eval()
        valid_loss = 0
        num_valid_steps = 0
        label_array_batch = []
        prob_array_batch = []
        with torch.no_grad():
            for ep, data in tqdm(enumerate(loader, 0)):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                labels = data['labels'].to(self.device, dtype = self.dtype)

                outputs = model(ids, mask, token_type_ids)

                if self.config['dataset']['task'] == 'regression':
                    labels = labels.reshape(-1, 1)

                loss = self.loss_function(outputs, labels)
                valid_loss += loss.item()

                if self.config['dataset']['task'] == 'classification':
                    probs = torch.softmax(outputs, dim = 1)[:,1]

                elif self.config['dataframe'] in ['FreeSolv', 'ESOL', 'Lipophilicity']: 
                    probs = self.normalizer.denorm(outputs)
                    labels = self.normalizer.denorm(labels)
            
                else:
                    probs = outputs

                if self.config['device'] == 'cpu':
                    label_array_batch.append(labels.numpy())
                    prob_array_batch.append(probs.detach().numpy())
                else:
                    label_array_batch.append(labels.cpu().numpy())
                    prob_array_batch.append(probs.detach().cpu().numpy())

                num_valid_steps += 1

            label_array_new = np.concatenate(label_array_batch, axis = 0)
            prob_array_new = np.concatenate(prob_array_batch, axis = 0)

            validation_loss = valid_loss/num_valid_steps
            print(f"Validation Loss: {validation_loss}")

            if self.config['dataset']['task'] == 'classification': 
                auc_score = roc_auc_score(label_array_new, prob_array_new)
                print(f"Validation AUC = {auc_score}")

                return validation_loss, auc_score

            elif self.config['dataset']['task'] == 'regression': 
                if self.config['dataframe'] in ['QM7', 'QM8', 'QM9']:
                    MAE_val = mean_absolute_error(label_array_new, prob_array_new)
                    print(f"Validation MAE = {MAE_val}")

                    return validation_loss, MAE_val

                else:
                    RMSE_val = mean_squared_error(label_array_new, prob_array_new, squared = False)
                    print(f"Validation RMSE = {RMSE_val}")

                    return validation_loss, RMSE_val
                
    def test(self, model, loader):
        model_path = os.path.join(self.log_dir, 'model.pth')
        state_dictionary = torch.load(model_path)
        model.load_state_dict(state_dictionary)
        print('Model Loaded Successfully')

        model.eval()
        prob_array_batch = []
        label_array_batch = []
        test_loss = 0
        num_test_steps = 0

        with torch.no_grad():
            for ep, data in tqdm(enumerate(loader, 0)): 
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                labels = data['labels'].to(self.device, dtype = self.dtype)

                outputs = model(ids, mask, token_type_ids)

                if self.config['dataset']['task'] == 'regression':
                    labels = labels.reshape(-1, 1)

                loss = self.loss_function(outputs, labels)
                test_loss += loss.item()

                if self.config['dataset']['task'] == 'classification':
                    probs = torch.softmax(outputs, dim = 1)[:,1]

                elif self.config['dataframe'] in ['FreeSolv', 'ESOL', 'Lipophilicity']:
                    probs = self.normalizer.denorm(outputs)
                    labels = self.normalizer.denorm(labels)

                else:
                    probs = outputs

                if self.config['device'] == 'cpu':
                    label_array_batch.append(labels.numpy())
                    prob_array_batch.append(probs.detach().numpy())
                else:
                    label_array_batch.append(labels.cpu().numpy())
                    prob_array_batch.append(probs.detach().cpu().numpy())

                num_test_steps += 1

            label_array_new = np.concatenate(label_array_batch, axis = 0)
            prob_array_new = np.concatenate(prob_array_batch, axis = 0)

            test_loss = test_loss/num_test_steps
            print(f"Test Loss: {test_loss}")

            if self.config['dataset']['task'] == 'classification':
                auc_score = roc_auc_score(label_array_new, prob_array_new)

                return test_loss, auc_score

            elif self.config['dataset']['task'] == 'regression': 
                if self.config['dataframe'] in ['QM7', 'QM8', 'QM9']:
                    MAE_test = mean_absolute_error(label_array_new, prob_array_new)

                    return test_loss, MAE_test

                else:
                    RMSE_test = mean_squared_error(label_array_new, prob_array_new, squared = False) 

                    return test_loss, RMSE_test
    
    def run(self):
        
        if self.config['model'] == 'bert':
            configuration = BertConfig(**self.config['model_bert'])
            model = BertClass(configuration).to(self.device)

        elif self.config['model'] == 'roberta':
            configuration = RobertaConfig(**self.config['model_roberta'])
            model = RobertaClass(configuration).to(self.device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr = float(self.config['learning_rate']))

        best_validate_auc = 0.0
        best_validate_regression = np.inf
        best_loss = np.inf
        early_stopping_counter = 0

        for epoch in range(self.config['epochs']): 
            auc_vals = []
            regression_vals = []
            validation_loss = []

            for target in target_list:
                print(f'For epoch = {epoch + 1} and column = {target}')

                self.normalizer = None
                if config['dataframe'] in ['FreeSolv', 'ESOL', 'Lipophilicity']:
                    self.normalizer = Normalizer(self.dataset[target])
                    self.dataset[target] = self.normalizer.norm(self.dataset[target])

                train_loader, validate_loader, _ = Data_Split(self.dataset, target, self.tokenizer, \
                                                                self.config['max_length'], \
                                                                **self.config['dataset']).get_data_loaders()
                torch.cuda.empty_cache()
                self.train(model, train_loader, optimizer)

                if config['dataset']['task'] == 'classification':
                    valid_loss, valid_auc = self.validate(model, validate_loader)
                    auc_vals.append(valid_auc)
                    validation_loss.append(valid_loss)

                elif config['dataset']['task'] == 'regression':
                    valid_loss, valid_regression = self.validate(model, validate_loader)
                    regression_vals.append(valid_regression)
                    validation_loss.append(valid_loss)
            
            mean_auc = np.mean(auc_vals)
            mean_regression = np.mean(regression_vals)
            mean_loss_per_epoch = np.mean(validation_loss)

            if self.config['dataset']['task'] == 'classification':
                if mean_auc > best_validate_auc:
                    best_validate_auc = mean_auc
                    test_path = self.log_dir + '/' + 'model.pth'
                    torch.save(model.state_dict(), test_path)

                    print(f"Validation AUC on best possible model = {best_validate_auc:1.3f}")

            elif self.config['dataset']['task'] == 'regression':
                if mean_regression < best_validate_regression:
                    best_validate_regression = mean_regression
                    test_path = self.log_dir + '/' + 'model.pth'
                    torch.save(model.state_dict(), test_path)

                    if self.config['dataframe'] in ['QM7', 'QM8', 'QM9']:
                        print(f"Validation MAE on best possible model = {best_validate_regression:1.3f}")
                    else:
                        print(f"Validation RMSE on best possible model = {best_validate_regression:1.3f}")
            
            if mean_loss_per_epoch < best_loss:
                best_loss = mean_loss_per_epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print(f'Early stopping count = {early_stopping_counter + 1}')

            if early_stopping_counter > self.config['early_stopping']:
                print(f'Early stopping triggered at epoch {epoch}!')
                break

        auc_test = []
        regression_test = []
        for target in target_list:
            print(f'label = {target}')
            _, _, test_loader = Data_Split(self.dataset, target, self.tokenizer, self.config['max_length'], \
                                                                **self.config['dataset']).get_data_loaders()

            if self.config['dataset']['task'] == 'classification':
                test_loss, test_auc = self.test(model, test_loader)
                auc_test.append(test_auc)

            elif self.config['dataset']['task'] == 'regression':
                test_loss, test_regression = self.test(model, test_loader)
                regression_test.append(valid_regression)

        mean_auc_test = np.mean(auc_test)
        mean_regression_test = np.mean(regression_test)

        if self.config['dataset']['task'] == 'classification':
            self.roc_auc = mean_auc_test
            print(f'Test AUC = {self.roc_auc:1.3f}')

        elif self.config['dataset']['task'] == 'regression': 
            if self.config['dataframe'] in ['QM7', 'QM8', 'QM9']:
                self.mae = mean_regression_test
                print(f"Test MAE = {self.mae:1.3f}")
            else:
                self.rmse = mean_regression_test
                print(f"Test RMSE = {self.rmse:1.3f}")

def main(config):
    dataset = Data_Processor(config['dataframe']).data_process()
    finetune = FineTune(dataset, config, tokenizer_path)
    finetune.run()
    
    if config['dataset']['task'] == 'classification':
        return finetune.roc_auc

    elif config['dataset']['task'] == 'regression':
        if config['dataframe'] in ['QM7', 'QM8', 'QM9']:
            return finetune.mae
        else:
            return finetune.rmse

if __name__ == '__main__':
    location = os.getcwd()

    with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
        config = yaml.safe_load(config_file)

    if config['model'] == 'bert':
        path = os.path.join(location, 'bert')
        tokenizer_path = os.path.join(path, 'vocab.txt')

    elif config['model'] == 'roberta':
        path = os.path.join(location, 'roberta')
        tokenizer_path = os.path.join(path, 'tokenizer_folder')

    if config['dataframe'] == 'BBBP':
        config['dataset']['task'] == 'classification'
        target_list = ['p_np']
        
    elif config['dataframe'] == 'SIDER':
        config['dataset']['task'] == 'classification'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif config['dataframe'] == 'ClinTox':
        config['dataset']['task'] == 'classification'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['dataframe'] == 'Tox21':
        config['dataset']['task'] == 'classification'
        target_list = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]

    elif config['dataframe'] == 'BACE':
        config['dataset']['task'] == 'classification'
        target_list = ['Class']

    elif config['dataframe'] == 'HIV':
        config['dataset']['task'] == 'classification'
        target_list = ['HIV_active']

    elif config['dataframe'] == 'MUV':
        config['dataset']['task'] == 'classification'
        target_list = [
            'MUV-692', 'MUV-846', 'MUV-859', 
            'MUV-644', 'MUV-548', 'MUV-600', 'MUV-810', 
            'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 
            'MUV-733', 'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['dataframe'] == 'ESOL':
        config['dataset']['task'] == 'regression'
        target_list = ["measured log solubility in mols per litre"]

    elif config['dataframe'] == 'FreeSolv':
        config['dataset']['task'] == 'regression'
        target_list = ["expt"]

    elif config['dataframe'] == 'Lipophilicity':
        config['dataset']['task'] == 'regression'
        target_list = ["exp"]

    elif config['dataframe'] == 'QM7':
        config['dataset']['task'] == 'regression'
        target_list = ["u0_atom"]

    elif config['dataframe'] == 'QM8':
        config['dataset']['task'] == 'regression'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    elif config['dataframe'] == 'QM9':
        config['dataset']['task'] == 'regression'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
    
    result = main(config)
