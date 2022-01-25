# Utility
import string
import numpy as np
import os

# Hugging Face library
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# Transformers Utility
from transformers_utils import tokenize_and_align_labels, compute_metrics

# Evaluation Utility
from utils import ner_classification_report


import transformers
print('Hugging Face Transformers Version :', transformers.__version__)

import tokenizers
print('Hugging Face Tokenizers Version :', tokenizers.__version__)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", help="Configuration Name")
parser.add_argument("--model_path", help="Path to store the output model")
parser.add_argument("--model_name", help="Name of the output model")
args = parser.parse_args()


CONFIG_NAME = args.config_name
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name
# Set Parameter and Hyper Parameter
model_checkpoint = "Geotrend/bert-base-th-cased" #หลักๆเราจะใช้ language model ของ Geotrend ภาษาไทย
batch_size = 4#16 #gpu เราบน colab น่าจะได้ batchsize แค่16



print('=====================================')
print('CONFIG_NAME \t\t= ', CONFIG_NAME)
print('MODEL_PATH \t\t= ', MODEL_PATH)
print('MODEL_NAME \t\t= ', MODEL_NAME)
print('=========  Hyperparameters  =========')
print('model_checkpoint \t= ', model_checkpoint)
print('batch_size \t\t= ', batch_size)
print('=====================================')


# Load dataset
datasets = load_dataset("semrolebank", CONFIG_NAME)
print('Dataset : ', datasets)
print('Sample  : ', datasets['validation'][0]['tokens'])

# Sanity check : There must be no empty tokens.
for s in datasets['validation']:
    for w in s['tokens']:
        if w == '':
            raise Exception('Word token is empty')


label_list = datasets["train"].features["role_tags"].feature.names
print('Label List : ', label_list)

# Load tokenizer (for transformers) and re-tokenize words accordingly as well as their labels
print (f"\nTokenization started ...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
print (f"Tokenization completed ...\n")

# Load pretrained model
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# Config model
args = TrainingArguments(
    MODEL_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

# Define data collator (for padding sentences)
data_collator = DataCollatorForTokenClassification(tokenizer)
# Define metric
metric = load_metric("seqeval")

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=lambda x: compute_metrics(x, metric, label_list)
)

model_filepath = MODEL_PATH + MODEL_NAME
if os.path.isfile(model_filepath):
    # Load Model
    print (f"\nTrained Model exists, Load model from {model_filepath}.")
    # retreive the saved model 
    trainer = AutoModelForTokenClassification.from_pretrained(model_filepath, local_files_only=True)
    trainer.to('cuda')
    print(f'Load Trained Model from {model_filepath} completed ...\n')
else:
    print('Training Model started ...')
    trainer.train()
    print('Training Model completed ...')

    trainer.evaluate()

    trainer.save_model(model_filepath)
    print(f'\nSave Trained Model Weights at {model_filepath} ...\n')

# Prediction on validation data
print('\nPrediction of validation data started ...')
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)
print('Prediction of validation data completed ...\n')

true_predictions = [
    [label_list[p].replace("_", "-") for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l].replace("_", "-") for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)

print('Align labels to original tokens ...\n')
y_pred = []
y_true = []
lbl_list = np.array(label_list)
for i, example in enumerate(tokenized_datasets["validation"]):
    example_token = tokenizer.convert_ids_to_tokens(example['input_ids'])
    j = 0
    pred_lbl = []
    
    gt_token = datasets['validation'][i]['tokens']
    
    for w, l in zip(example_token[1:-1], true_predictions[i]):
        if j >= len(gt_token):
            if w[0] not in string.punctuation and not w.isnumeric():
                print(w, example['tokens'], example_token[1:-1])
                raise Exception('Unexpected token')
        elif gt_token[j].startswith(w) or w == '[UNK]':
            pred_lbl.append(l)
            j = j+1
    y_pred.append(pred_lbl)
    
    gt_lbl = lbl_list[example['role_tags']].tolist()
    y_true.append(gt_lbl)
    
    if len(pred_lbl) != len(gt_lbl):
        print(w, datasets['validation'][i]['tokens'], example, example_token)
        print(len(pred_lbl), len(gt_lbl))
    assert(len(pred_lbl) == len(gt_lbl))
    
print(ner_classification_report(y_true, y_pred))