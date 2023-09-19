# GPT-MolBERTa (Text-based molecular property prediction)
## Summary
<img src='https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/assets/112913550/54a810e7-3dc2-46c1-94e2-0a1774f30b21' width="250" height='400' align="right">
GPT-MolBERTa is a text based molecular property prediction model. Instead of using SMILES, it utilizes a text-based representation of the SMILES molecules containing information such as atom types, elemental properties, geometry and some other relevant information. GPT-MolBERTa was pretrained with these descriptions and finetuned on selected datasets from the MoleculeNet benchmark.

## Details
* Provides text-based molecular datasets for MoleculeNet datasets

* Compares the effectiveness for BERT and RoBERTa models in property prediction

* Visualizes the attention map of the input data, displaying how GPT-MolBERTa analyzes the data

* Achieves strong performance in the regression tasks of MoleculeNet

<img src = 'https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/assets/112913550/b054f041-4c90-49e6-a204-3f9970025b1a' width='750'>

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
Any new dataset should be placed in the dataset folder, with the SMILES column named as 'smiles'. In the config_finetune.yaml file, the name of the dataset should be specified within the 'dataframe' argument. An example dataset is provided under the dataset folder.

2. Processed Textual Dataset
The datasets used for this project are nine official datasets from MoleculeNet with text descriptions added. The datasets are provided in this link: [Datasets](https://drive.google.com/file/d/1ECiSlUT8yvJBSErjR9f_dLV_2r6PxCQe/view?usp=drive_link). The zip file should be unzipped and placed in GPT-MolBERTa folder.

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

The pretrained RoBERTa model is found in this link: [Pretrained RoBERTa](https://drive.google.com/file/d/1_vKE6Rb9A7PU0PVTu3u1XlaqoPXsp9sJ/view?usp=drive_link). Remember to place this model in the roberta folder.
The pretrained BERT model is found in this link: [Pretrained BERT](https://drive.google.com/file/d/1b39vo6OZIa76VacZL50oT7mbO6Yue6l6/view?usp=drive_link). Remember to place this model in the bert folder.

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
