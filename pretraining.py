from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import BertTokenizer,  BertConfig, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
import torch
from rouge import Rouge
import logging
logging.basicConfig(level = logging.ERROR)
import os
import yaml
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_metric

location = os.getcwd() # /home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

if config['model'] == 'bert': # pretrain this as well
    path = os.path.join(location, 'bert')
    filepath = os.path.join(path, 'pretrain_data_bert.txt')

    vocab_file_directory = os.path.join(path, 'vocab.txt')

    tokenizer = BertTokenizer.from_pretrained(vocab_file_directory)

    configuration = BertConfig(**config['model_bert'])
    
    model = BertForMaskedLM(configuration)

    print(f'Number of parameters: {model.num_parameters()}')

elif config['model'] == 'roberta':
    path = os.path.join(location, 'roberta')
    train_file = os.path.join(path, 'pretrain_data_roberta_train.txt')
    eval_file = os.path.join(path, 'pretrain_data_roberta_eval.txt')

    vocab_file_directory = os.path.join(path, 'tokenizer_folder')

    # tokenizer = RobertaTokenizer.from_pretrained(vocab_file_directory)
    tokenizer = RobertaTokenizer(
        os.path.join(vocab_file_directory, 'vocab.json'),
        os.path.join(vocab_file_directory, 'merges.txt')
    )

    configuration = RobertaConfig(**config['model_roberta'])
    
    model = RobertaForMaskedLM(configuration)

    print(f'Number of parameters: {model.num_parameters()}')

train_dataset = LineByLineTextDataset( # Check this again
    tokenizer = tokenizer,
    file_path = train_file,
    block_size = 128
)

eval_dataset = LineByLineTextDataset( 
    tokenizer = tokenizer,
    file_path = eval_file,
    block_size = 128
)

print(f'No. of lines: {len(train_dataset) + len(eval_dataset)}')

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)
def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    rouge = Rouge()
    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )

    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

training_args = TrainingArguments(
    output_dir = path,
    overwrite_output_dir = True,
    num_train_epochs = 7,
    do_eval = True,
    evaluation_strategy = "steps",
    eval_steps = 100,
    per_device_train_batch_size = 30,
    per_device_eval_batch_size = 6,
    # eval_accumulation_steps  = 1,
    save_steps = 1000,
    save_total_limit = 5
)
# 60
## Keep note of the loss (screenshot the loss), do validation set (5%) for 172k pre-training
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator, 
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model(path)