import numpy as np
import pandas as pd
import os
import json

df = pd.read_csv(path_to_dataset)
length = len(df['smiles'])

path = os.getcwd() #/home/suryabalaji/GPT_MolBERTa
number = np.arange(1, length + 1)
directory = os.path.join(path, 'text_files_' + 'var')
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
df.to_csv(path + '/' + var + ' dataset.csv', sep = ',', encoding = 'utf-8')