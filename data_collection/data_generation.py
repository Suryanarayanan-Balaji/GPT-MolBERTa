import concurrent.futures
import pandas as pd
import openai
import backoff
import os

path = os.getcwd() #/home/suryabalaji/GPT_MolBERTa
api_key_path = path + '/' + 'GPT_API_Key.txt'

openai.organization = 'org-HwfChX2caOsYiQUzvp1XmmNU'
openai.api_key_path = api_key_path

MODEL = 'gpt-3.5-turbo'
filename = os.path.join(path, 'text_files_' + 'var')

def generate_completion(name, value):
    response = completions_with_backoff(
        model = MODEL,
        messages = [
            {"role": "system", "content": 'You are able to generate important and verifiable features about molecular SMILES'},
            {"role": "user", "content": f"Generate a description about the following SMILES molecule {value}"},
        ],
        temperature = 0,
        max_tokens = 2048
    )
    with open(filename + "/" + str(name) + '.txt', 'w') as f:
        f.write(str(response))
        
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)    
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def data_gen(idx):
    df = pd.read_csv(path_to_dataset)
    x_data = df['smiles'][idx:]    
    
    mapping = {}
    for index, row in pd.DataFrame(x_data).iterrows():
        smiles = row['smiles']
        name = 'Molecule' + ' ' + str(index + 1)
        mapping[name] = smiles    
    
    max_threads = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers = max_threads) as executor:
        futures = [executor.submit(generate_completion, name, value) for name, value in mapping.items()]
        concurrent.futures.wait(futures)

length = len(pd.read_csv(path_to_dataset)['smiles'])

current_number = 0

while current_number < length:

    try:
        data_gen(current_number)
    except:
        for root, dirs, files in os.walk(filename, topdown=False):
            current_number = len(files)
