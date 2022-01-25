import tensorflow as tf
import keras
print('Tensorflow Version :', tf.__version__)
print('Keras Version :', keras.__version__)

# Save / Load File
import dill
import pickle

# Plot Graph
import matplotlib.pyplot as plt

# Load Vectors
from gensim.models import KeyedVectors

# Utility
import numpy as np
import time
import os

# Model Utility
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# Evaluation Utility
from utils import ner_classification_report

# Keras Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.utils import to_categorical
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint


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

# Set Parameter and Hyper Parameter
max_len = 250
max_len_char = 30

character_LSTM_unit = 32
char_embedding_dim = 32
main_lstm_unit = 256 ## Bidirectional 256 + 256 = 512
lstm_recurrent_dropout = 0.5

train_batch_size = 32
train_epochs = 40



print('=====================================')
print('TRAIN_PATH \t\t= ', TRAIN_PATH)
print('VALIDATION_PATH \t= ', VALIDATION_PATH)
print('MODEL_PATH \t\t= ', MODEL_PATH)
print('MODEL_NAME \t\t= ', MODEL_NAME)
print('Word_Embed__MODEL_PATH \t= ', W_MODEL_PATH)
print('Dict_MODEL_PATH \t= ', Dict_MODEL_PATH)
print('=========  Hyperparameters  =========')
print('max_len \t\t= ', max_len)
print('max_len_char \t\t= ', max_len_char, '\n')
print('character_LSTM_unit \t= ', character_LSTM_unit)
print('char_embedding_dim \t= ', char_embedding_dim)
print('main_lstm_unit \t\t= ', main_lstm_unit)
print('lstm_recurrent_dropout \t= ', lstm_recurrent_dropout, '\n')
print('train_batch_size \t= ', train_batch_size)
print('train_epochs \t\t= ', train_epochs)
print('=====================================')

os.makedirs(Dict_MODEL_PATH, exist_ok=True)

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

word_list.append("pad")
word_list.append("unknown") #Special Token for Unknown words ("UNK")
ner_list.append("pad")

all_words = sorted(set(word_list))
all_ner = sorted(set(ner_list))
all_thai2dict = sorted(set(thai2dict))

word_to_ix = dict((c, i) for i, c in enumerate(all_words)) #convert word to index 
ner_to_ix = dict((c, i) for i, c in enumerate(all_ner)) #convert ner to index
thai2dict_to_ix = dict((c, i) for i, c in enumerate(thai2dict)) #convert thai2fit to index 

ix_to_word = dict((v,k) for k,v in word_to_ix.items()) #convert index to word
ix_to_ner = dict((v,k) for k,v in ner_to_ix.items())  #convert index to ner
ix_to_thai2dict = dict((v,k) for k,v in thai2dict_to_ix.items())  #convert index to thai2fit

n_word = len(word_to_ix)
n_tag = len(ner_to_ix)
n_thai2dict = len(thai2dict_to_ix)
print('No. of unique words : ', n_word)
print('No. of tags : ', n_tag)
print('No. of known words : ', n_thai2dict)
print('Semantic role tag : ', ner_to_ix)


# Preprocess Character
chars = set([w_i for w in thai2dict for w_i in w])
char2idx = {c: i + 5 for i, c in enumerate(chars)}

char2idx["pad"] = 0
char2idx["unknown"] = 1
char2idx[" "] = 2

char2idx["$"] = 3
char2idx["#"] = 4
char2idx["!"] = 5
char2idx["%"] = 6
char2idx["&"] = 7
char2idx["*"] = 8
char2idx["+"] = 9
char2idx[","] = 10
char2idx["-"] = 11
char2idx["."] = 12
char2idx["/"] = 13
char2idx[":"] = 14
char2idx[";"] = 15
char2idx["?"] = 16
char2idx["@"] = 17
char2idx["^"] = 18
char2idx["_"] = 19
char2idx["`"] = 20
char2idx["="] = 21
char2idx["|"] = 22
char2idx["~"] = 23
char2idx["'"] = 24
char2idx['"'] = 25

char2idx["("] = 26
char2idx[")"] = 27
char2idx["{"] = 28
char2idx["}"] = 29
char2idx["<"] = 30
char2idx[">"] = 31
char2idx["["] = 32
char2idx["]"] = 33

n_chars = len(char2idx)
print('No. of unique characters : ', n_chars)

# Save Dictionary for Character and NER
with open(Dict_MODEL_PATH+'chardict.pickle', 'wb') as chardict:
    pickle.dump(char2idx, chardict)

with open(Dict_MODEL_PATH+'nerdict.pickle', 'wb') as nerdict:
    pickle.dump(ner_to_ix, nerdict)
    
print('\nSave Dictionary for Character and Semantic Role Tag completed ...\n')


# Mapping Function 
def prepare_sequence_word(input_text):
    idxs = list()
    for word in input_text:
        if word in thai2dict:
            idxs.append(thai2dict_to_ix[word])
        else:
            idxs.append(thai2dict_to_ix["unknown"]) #Use UNK tag for unknown word
    return idxs

def prepare_sequence_target(input_label):
    idxs = [ner_to_ix[w] for w in input_label]
    return idxs


# Split word and label
input_sent =[ [ word[0] for word in sent]for sent in train_sents ] #words only
train_targets =[ [ word[1] for word in sent]for sent in train_sents ] #NER only

input_test_sent =[ [ word[0] for word in sent]for sent in test_sents ] #words only
test_targets =[ [ word[1] for word in sent]for sent in test_sents ] #NER only

print('Prepare Training Dataset started ...')
## Word Training
X_word_tr = [prepare_sequence_word(s) for s in input_sent]
X_word_tr = pad_sequences(maxlen=max_len, sequences=X_word_tr, value=thai2dict_to_ix["pad"], padding='post', truncating='post')

## Character Training
X_char_tr = []
for sentence in train_sents:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                if(sentence[i][0][j] in char2idx):
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                else:
                    word_seq.append(char2idx.get("unknown"))
            except:
                word_seq.append(char2idx.get("pad"))
        sent_seq.append(word_seq)
    X_char_tr.append(np.array(sent_seq))

## Sequence Label Training
y_tr = [prepare_sequence_target(s) for s in train_targets]
y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, value=ner_to_ix["pad"], padding='post', truncating='post')
y_tr = [to_categorical(i, num_classes=n_tag) for i in y_tr]
print('Prepare Training Dataset completed ...\n')

## Word Testing
X_word_te = [prepare_sequence_word(s) for s in input_test_sent]
X_word_te = pad_sequences(maxlen=max_len, sequences=X_word_te, value=thai2dict_to_ix["pad"], padding='post', truncating='post')

print('Prepare Validation Dataset started ...')
## Character Testing
X_char_te = []
for sentence in test_sents:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                if(sentence[i][0][j] in char2idx):
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                else:
                    word_seq.append(char2idx.get("unknown"))
            except:
                word_seq.append(char2idx.get("pad"))    
        sent_seq.append(word_seq)
    X_char_te.append(np.array(sent_seq))

## Sequence Label Testing
y_te = [prepare_sequence_target(s) for s in test_targets]
y_te = pad_sequences(maxlen=max_len, sequences=y_te, value=ner_to_ix["pad"], padding='post', truncating='post')
y_te = [to_categorical(i, num_classes=n_tag) for i in y_te]
print('Prepare Validation Dataset completed ...\n')

## Initial Keras Model
# Word Input
word_in = Input(shape=(max_len,), name='word_input_')

# Word Embedding Using Thai2Fit
word_embeddings = Embedding(input_dim=n_thai2dict,
                            output_dim=400,
                            weights = [thai2fit_weight],input_length=max_len,
                            mask_zero=False,
                            name='word_embedding', trainable=False)(word_in)

# Character Input
char_in = Input(shape=(max_len, max_len_char,), name='char_input')

# Character Embedding
emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim=char_embedding_dim, 
                           input_length=max_len_char, mask_zero=False))(char_in)

# Character Sequence to Vector via BiLSTM
char_enc = TimeDistributed(Bidirectional(LSTM(units=character_LSTM_unit, return_sequences=False, recurrent_dropout=lstm_recurrent_dropout)))(emb_char)


# Concatenate All Embedding
all_word_embeddings = concatenate([word_embeddings, char_enc])
all_word_embeddings = SpatialDropout1D(0.3)(all_word_embeddings)

# Main Model BiLSTM
main_lstm = Bidirectional(LSTM(units=main_lstm_unit, return_sequences=True,
                               recurrent_dropout=lstm_recurrent_dropout))(all_word_embeddings)
main_lstm = TimeDistributed(Dense(50, activation="relu"))(main_lstm)

# CRF
crf = CRF(n_tag)  # CRF layer
out = crf(main_lstm)  # output

# Model
model = Model([word_in, char_in], out)

model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()


model_filepath = MODEL_PATH+MODEL_NAME
if os.path.isfile(model_filepath):
    # Load Model
    print (f"\nTrained Model exists, Load model from {model_filepath}.")
    model.load_weights(model_filepath)
    print(f'Load Trained Model from {model_filepath} completed ...\n')
else:
    # Training Model
    print('Training Model started ...')
    filepath=MODEL_PATH+"checkpoint.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit([X_word_tr,
                         np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))
                         ],
                         np.array(y_tr),
                         batch_size=train_batch_size, epochs=train_epochs, verbose=1,callbacks=callbacks_list,
                         validation_data=(
                         [X_word_te,
                         np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))
                         ],
                         np.array(y_te))
                       )
    print('Training Model completed ...')
    
    # Plot Accuracy Graph
    hist = pd.DataFrame(history.history)

    plt.style.use("ggplot")

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].plot(hist["crf_viterbi_accuracy"], label="crf_viterbi_accuracy")
    axs[0].plot(hist["val_crf_viterbi_accuracy"], label="val_crf_viterbi_accuracy")
    axs[0].legend()


    # Plot Loss Graph
    hist = pd.DataFrame(history.history)

    # plt.style.use("ggplot")
    # plt.figure(figsize=(8,8))
    axs[1].plot(hist["loss"], label='loss')
    axs[1].plot(hist["val_loss"], label='val_loss')
    axs[1].legend()
    plt.show()

    save_filepath=MODEL_PATH+MODEL_NAME
    model.save_weights(save_filepath)
    print(f'\nSave Trained Model Weights at {save_filepath} ...\n')

# Prediction on validation data
print('\nPrediction of validation data started ...')
pred_model = model.predict([X_word_te,np.array(X_char_te).reshape((len(X_char_te),max_len, max_len_char))], verbose=1)

y_pred = []
y_true = []

for i in range(0,len(pred_model)):
    try:
        out = np.argmax(pred_model[i], axis=-1)
        true = np.argmax(y_te[i], axis=-1)
        revert_pred=[ix_to_ner[i] for i in out]
        revert_true=[ix_to_ner[i] for i in true]
        y_pred.append(revert_pred)
        y_true.append(revert_true)
    except:
        print (i)

print(ner_classification_report(y_true,y_pred))
    


