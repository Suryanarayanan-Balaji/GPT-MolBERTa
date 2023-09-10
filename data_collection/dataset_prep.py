import numpy as np
import pandas as pd
import os
import json
import yaml

location = os.getcwd() #/home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

path_to_dataset = os.path.join(location, 'datasets')
dataset_path = path_to_dataset + '/' + str(config['dataframe']) + '.csv'

df = pd.read_csv(dataset_path)
length = len(df['smiles'])
number = np.arange(1, length + 1)
directory = os.path.join(location, 'data_gen', 'text_files_' + config['dataframe'])
descriptions = []
num_tokens = []
for i in number:
    with open(directory + '/' + 'Molecule ' + str(i) + '.txt', 'r') as f:
        json_content = f.read()
        data = json.loads(json_content)
        message = data['choices'][0]['message']['content']
        num = data['usage']['completion_tokens']
        descriptions.append(message)
        num_tokens.append(num)

df.loc[:, 'Descriptions'] = descriptions
df.loc[:, 'count'] = num_tokens
df = df[df['count'] >= 100].reset_index(drop = True)
df.to_csv(location + '/data/' + str(config['dataframe']) + ' dataset.csv', sep = ',', encoding = 'utf-8')
