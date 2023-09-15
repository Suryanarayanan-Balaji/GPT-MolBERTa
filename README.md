# GPT-MolBERTa (Text-based molecular property prediction)

## Summary
GPT-MolBERTa is a text based molecular property prediction model. Instead of using SMILES, it utilizes a text-based representation of the SMILES molecules containing information such as atom types, elemental properties, geometry and some other relevant information. GPT-MolBERTa was pretrained with these descriptions and finetuned on selected datasets from the MoleculeNet benchmark. ![DreamShaper_v7_Molecules_description_generated_with_chatGPT_an_2](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/assets/112913550/3c147ccf-bef4-4d81-9fc7-36d3eb31bf7f, width = 100)
.


## Details
* Provides text-based molecular datasets for MoleculeNet datasets

* Compares the effectiveness for BERT and RoBERTa models in property prediction

* Visualizes the attention map of the input data, displaying how GPT-MolBERTa analyzes the data

* Achieves strong performance in the regression tasks of MoleculeNet

![GPTMolBerta](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/assets/112913550/b054f041-4c90-49e6-a204-3f9970025b1a, width = 100)

## Getting Started
Here are the required prerequisites which are required to be installed:

### Software Required 
openai 0.27.6 

backoff 2.2.1 

python 3.10.1 

tokenizers 0.11.4 

transformers 4.28.0 

rdkit 2023.3.2 

In addition to this, an OpenAI account along with a GPT API Key should also be created. The GPT API Key should be a text file in the GPT-MolBERTa folder.
## Installation of Repository

1. Clone the repository using the following command
```python
# Clone the repository of GPT-MolBERTa
git clone https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa.git
cd GPT-MolBERTa
```
### Dataset
1. Raw Text Data
Text descriptions can be generated and added to any dataset as required. The data generation process involves a CSV dataset, an OpenAI account, and a GPT API Key, and is outlined below:

  1.1 Run the data_generation.py script using the following command
   ```python
    python ./data_collection/data_generation.py
   ```
  1.2 If there are still missing text descriptions, run the data_generation_for_missing.py script using the following command
   ```python
   python ./data_collection/data_generation_for_missing.py
   ```
  1.3 Transfer all the descriptions from the folder generated in 1.2 to that generated in 1.1.
  
  1.4 Run dataset_prep.py to compile the dataset together.
   ```python
   python ./data_collection/dataset_prep.py
   ```
2. Processed Textual Dataset
The datasets used for this project are nine official datasets from MoleculeNet with text descriptions added. The datasets are provided in this link: [https://drive.google.com/file/d/1ECiSlUT8yvJBSErjR9f_dLV_2r6PxCQe/view?usp=drive_link](https://drive.google.com/file/d/1ECiSlUT8yvJBSErjR9f_dLV_2r6PxCQe/view?usp=drive_link). The zip file should be unzipped and placed in GPT-MolBERTa folder.

### Pretraining
If pretraining the model with other datasets, the following steps must be followed.
1. Tokenize the corpus by running tokenization.py using the following command.
```python
python tokenization.py
```
2. Pretrain the model using pretraining.py script with the following command.
```python
python pretraining.py
```
The hyperparameters for pretraining are found in the config_pretrain.yaml file.

### Finetuning
The training configurations for GPT-MolBERTa can be found in the config_finetune.yaml file.

During the finetuning process, GPT-MolBERTa creates and stores the model in the finetune folder. This folder is created automatically if it doesn't exist and serves as the storage location for the finetuned models. The command for finetuning is as follows.

 ```python
 python finetune.py
 ```
### Attention Score Analysis
The AttentionVisualizer repository offers a comprehensive toolkit for visualizing and analyzing attention scores. To effectively utilize this tool in conjunction with GPT-MolBERTa, you can integrate the finetuned Roberta encoder into the AttentionVisualizer package.

## Contact
For more information about the model checkpoints and datasets, please contact suryanab@andrew.cmu.edu
