# GPT-MolBERTa (Text-based molecular property prediction)

## Summary
GPT-MolBERTa is a text based molecular property prediction model. Instead of using SMILES, it utilizes a text-based representation of the SMILES molecules containing information such as atom types, elemental properties, geometry and some other relevant information. GPT-MolBERTa was pretrained with these descriptions and finetuned on selected datasets from the MoleculeNet benchmark ![DreamShaper_v7_Molecules_description_generated_with_chatGPT_an_2](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/assets/112913550/3c147ccf-bef4-4d81-9fc7-36d3eb31bf7f)
.


## Details
* Provides text-based molecular datasets for MoleculeNet datasets

* Compares the effectiveness for BERT and RoBERTa models in property prediction

* Visualizes the attention map of the input data, displaying how GPT-MolBERTa analyzes the data

* Achieves strong performance in the regression tasks of MoleculeNet
[GPTMolBerta.pdf](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/files/12598549/GPTMolBerta.pdf)

## Getting Started

### Software Required (Add some more)
openai 0.27.6 \\
backoff 2.2.1 \\
python 3.10.1 \\
tokenizers 0.11.4 \\
transformers 4.28.0 \\
rdkit 2023.3.2 \\

### Installation of Repository

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
3. Processed Textual Dataset
The datasets used for this project are nine official datasets from MoleculeNet with text descriptions added. The datasets are provided in this link: https://drive.google.com/file/d/1ECiSlUT8yvJBSErjR9f_dLV_2r6PxCQe/view?usp=drive_link. The zip file should be unzipped and placed in GPT-MolBERTa folder.

### Finetuning
1. 

