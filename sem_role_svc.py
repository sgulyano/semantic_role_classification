# Save / Load File
import dill
import pickle

# Load Vectors
from gensim.models import KeyedVectors

# Utility
import numpy as np
import time
import os

# Model Utility
from sklearn.model_selection import train_test_split
from sklearn import svm
from tqdm import tqdm
import pandas as pd

# Evaluation Utility
from utils import ner_classification_report

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", help="Path to training data")
parser.add_argument("--validation_data", help="Path to validation data")
parser.add_argument("--model_path", help="Path to store the output model")
parser.add_argument("--model_name", help="Name of the output model")
parser.add_argument("--w_model_path", help="Path to weight of word embedding model (Thai2Fit)")
args = parser.parse_args()


TRAIN_PATH = args.train_data
VALIDATION_PATH = args.validation_data
MODEL_PATH = args.model_path
MODEL_NAME = args.model_name
W_MODEL_PATH = args.w_model_path
Dict_MODEL_PATH = f'{MODEL_PATH}/dictionary/'

os.makedirs(Dict_MODEL_PATH, exist_ok=True)

print('=====================================')
print('TRAIN_PATH \t\t= ', TRAIN_PATH)
print('VALIDATION_PATH \t= ', VALIDATION_PATH)
print('MODEL_PATH \t\t= ', MODEL_PATH)
print('MODEL_NAME \t\t= ', MODEL_NAME)
print('Word_Embed__MODEL_PATH \t= ', W_MODEL_PATH)
print('Dict_MODEL_PATH \t= ', Dict_MODEL_PATH)
print('=====================================')


# Load raw dataset (NER)
with open(TRAIN_PATH, 'rb') as file:
    train_sents = dill.load(file)
    
with open(VALIDATION_PATH, 'rb') as file:
    test_sents = dill.load(file)

print('Train sentences = ', len(train_sents))
print('Test sentences  = ', len(test_sents))
print('Sample sentence : ', train_sents[20])

# Load Thai2Fit Word Embedding
# load Binary file of thai2fit (0.32) train wikipedia using ULMFit model
# credit: https://github.com/cstorm125/thai2fit
thai2fit_model = KeyedVectors.load_word2vec_format(W_MODEL_PATH+'thai2vecNoSym.bin',binary=True)
thai2fit_weight = thai2fit_model.vectors
print('\nLoad Thai2Fit model completed ...\n')

# Preprocess Word
word_list=[]
ner_list=[]
thai2dict = {}

for sent in train_sents:
    for word in sent:
        word_list.append(word[0])
        ner_list.append(word[1])
        
for word in thai2fit_model.index2word:
    thai2dict[word] = thai2fit_model[word]

word_list.append("unknown") #Special Token for Unknown words ("UNK")

all_words = sorted(set(word_list))
all_ner = sorted(set(ner_list))
all_thai2dict = sorted(set(thai2dict))

ner_to_ix = dict((c, i) for i, c in enumerate(all_ner)) #convert ner to index
ix_to_ner = dict((v,k) for k,v in ner_to_ix.items())  #convert index to ner

n_word = len(all_words)
n_tag = len(all_ner)
n_thai2dict = len(all_thai2dict)
print('No. of unique words : ', n_word)
print('No. of tags : ', n_tag)
print('No. of known words : ', n_thai2dict)
print('Semantic role tag : ', ner_to_ix)

# Save Dictionary for NER
with open(Dict_MODEL_PATH+'semroledict.pickle', 'wb') as nerdict:
    pickle.dump(ner_to_ix, nerdict)
print('\nSave Dictionary for Semantic Role Tag completed ...\n')

# Mapping Function 
def prepare_sequence_target(input_label):
    idxs = [ner_to_ix[w] for w in input_label]
    return idxs

def prepare_sequence_vector(input_text):
    idxs = list()
    for word in input_text:
        if word in thai2dict:
            idxs.append(thai2dict[word])
        else:
            idxs.append(thai2dict["unknown"]) #Use UNK tag for unknown word
    return idxs

def add_sequence_previous(input_sent):
    vec_list = [thai2dict["unknown"]] + input_sent
    feat_vec = [np.hstack(vec_list[i:i+2]) for i in range(len(input_sent))]
    return feat_vec

# Split word and label
input_sent =[ [ word[0] for word in sent]for sent in train_sents ] #words only
train_targets =[ [ word[1] for word in sent]for sent in train_sents ] #NER only

input_test_sent =[ [ word[0] for word in sent]for sent in test_sents ] #words only
test_targets =[ [ word[1] for word in sent]for sent in test_sents ] #NER only

# Prepare Training Dataset
X_word_vec = [add_sequence_previous(prepare_sequence_vector(s)) for s in input_sent]
X_word_tr = np.vstack(X_word_vec)

y_tr = [prepare_sequence_target(s) for s in train_targets]
y_word_tr = np.hstack(y_tr)

print('Size of X_train : ', X_word_tr.shape)
print('Size of y_train : ', y_word_tr.shape)
print('Prepare Training Dataset completed ...\n')

# Prepare Testing Dataset
X_word_te = [add_sequence_previous(prepare_sequence_vector(s)) for s in input_test_sent]

y_word_te = [prepare_sequence_target(s) for s in test_targets]

print('No. of y_test : ', len(X_word_te))
print('No. of y_test : ', len(y_word_te))
print('Prepare Testing Dataset completed ...\n')


model_filepath = MODEL_PATH+MODEL_NAME
if os.path.isfile(model_filepath):
    print (f"\nTrained Model exists, Load model from {model_filepath}.")
    clf = pickle.load(open(model_filepath, 'rb'))
    print(f'Load Trained Model from {model_filepath} completed ...\n')
    
else:
    # Training Model
    print('Training Model started ...')
    clf = svm.SVC(decision_function_shape='ovo', random_state=0)
    clf.fit(X_word_tr, y_word_tr)
    print('Training Model completed ...')

    pickle.dump(clf, open(model_filepath, 'wb'))
    print(f'\nSave Trained Model at {model_filepath} ...\n')
    
# Prediction on validation data
print('\nPrediction of validation data started ...')
y_word_pr = [clf.predict(x) for x in X_word_te]
print('\nPrediction of validation data completed ...')

# convert tag index to tag string
y_pred = []
y_true = []

for j in range(0,len(y_word_te)):
    revert_pred=[ix_to_ner[i] for i in y_word_pr[j]]
    revert_true=[ix_to_ner[i] for i in y_word_te[j]]
    y_pred.append(revert_pred)
    y_true.append(revert_true)

print(ner_classification_report(y_true,y_pred))
    



