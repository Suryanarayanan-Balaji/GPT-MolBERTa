# GPT-MolBERTa (Text-based molecular property prediction)

## Summary
GPT-MolBERTa is a text based molecular property prediction model. Instead of using SMILES, it utilizes a text-based representation of the SMILES molecules containing information such as atom types, elemental properties, geometry and some other relevant information. GPT-MolBERTa was pretrained with these descriptions and finetuned on selected datasets from the MoleculeNet benchmark ![DreamShaper_v7_Molecules_description_generated_with_chatGPT_an_2](https://github.com/Suryanarayanan-Balaji/GPT-MolBERTa/assets/112913550/3c147ccf-bef4-4d81-9fc7-36d3eb31bf7f)
.


## Details
* Provides text-based molecular datasets for MoleculeNet datasets

* Compares the effectiveness for BERT and RoBERTa models in property prediction

* Visualizes the attention map of the input data, displaying how GPT-MolBERTa analyzes the data

* Achieves strong performance in the regression tasks of MoleculeNet

## Getting Started

### Software Required
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
1. Processed Textual Dataset
The datasets used for this project are nine official datasets from MoleculeNet with text descriptions added. The datasets are provided in this link: https://drive.google.com/file/d/1ECiSlUT8yvJBSErjR9f_dLV_2r6PxCQe/view?usp=drive_link. The zip file should be unzipped and placed in GPT-MolBERTa folder.

