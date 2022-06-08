# -*- coding: utf-8 -*-
"""

Steps:
*   Download the corpora and split it in training and test sets, structuring a dataframe.
*   Embed the words using GloVe embeddings
*   Create a baseline model, using a simple neural architecture
*   Experiment doing small modifications to the model
*   Evaluate your best model

**Corpora**:
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip

**Splits**: documents 1-100 are the train set, 101-150 validation set, 151-199 test set.

**Baseline**: two layers architecture: a Bidirectional LSTM and a Dense/Fully-Connected layer on top.

**Modifications**: experiment using a GRU instead of the LSTM, adding an additional LSTM layer, and using a CRF in addition to the LSTM. Each of this change must be done by itself (don't mix these modifications).

**Training and Experiments**: all the experiments must involve only the training and validation sets.

**Evaluation**: in the end, only the best model of your choice must be evaluated on the test set. The main metric must be F1-Macro computed between the various part of speech (without considering punctuation classes).

"""

# system packages
import os
import shutil
import sys
import math

# data and numerical management packages
import pandas as pd
import numpy as np

# useful during debugging (progress bars)
from tqdm import tqdm

# custom import
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, Embedding, TimeDistributed, Masking, GRU



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

import gensim
import gensim.downloader as gloader

from tf2crf import CRF, ModelWithCRFLoss

"""## Download and Extraction"""

import nltk
nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank

pos_tags = []
for sent in dependency_treebank.tagged_sents():
  for element in sent:
    pos_tags.append(element[1])

pos_tags = list(set(pos_tags))
#pos_tags.insert(0, "PAD")
tags_encoder = OneHotEncoder()
tags_binary = tags_encoder.fit(np.array(pos_tags).reshape(-1, 1))

def build_splitted_dataset(start_index, stop_index):
  df_sents = []
  df_tags = []
  for sent in dependency_treebank.tagged_sents()[start_index:stop_index]:
    curr_sent = []
    curr_tags = []
    # sent is a tuple (value, tag)
    for element in sent:
      curr_sent.append(element[0])
      curr_tags.append(tags_binary.transform(np.array(element[1]).reshape(-1, 1)).toarray()[0])

    df_sents.append(curr_sent)
    df_tags.append(curr_tags)

  return df_sents, df_tags

# Define indexes of the dataset based on files
start_index_train = 0
end_index_train = 1962

start_index_validate = 1963
end_index_validate = 3262

start_index_test = 3263
end_index_test = 3914

X_train, Y_train = build_splitted_dataset(start_index_train, end_index_train)
print(len(X_train), len(Y_train))
X_validate, Y_validate = build_splitted_dataset(start_index_validate, end_index_validate)
X_test, Y_test = build_splitted_dataset(start_index_test, end_index_test)

"""## Create vocabularies"""

def create_vocabulary(dataset):
  words = []
  for sentence in dataset:
    for word in sentence:
      words.append(word)

  return list(set(words))

train_val_vocab = create_vocabulary(X_train+X_validate)
test_vocab = create_vocabulary(X_test)

# Store vocab in a dict, associated with a numerical index
train_val_word2idx = { word[1]: word[0] for word in enumerate(train_val_vocab) }
train_val_idx2word = { v: k for k, v in train_val_word2idx.items() }

test_word2idx = { word[1]: word[0] for word in enumerate(test_vocab) }
test_idx2word = { v: k for k, v in test_word2idx.items() }

"""## Encode dataset"""

def encode_dataset(dataset, word2idx):
  encoded_dataset = []
  for sente in dataset:
    curr_sente = []
    for word in sente:
      curr_sente.append(word2idx[word])
    encoded_dataset.append(curr_sente)
  return encoded_dataset

#Transforming train, val, test word lists in id lists
X_train_encoded = encode_dataset(X_train, train_val_word2idx)
X_validate_encoded = encode_dataset(X_validate, train_val_word2idx)
X_test_encoded = encode_dataset(X_test, test_word2idx)

"""## Embedding Model extraction"""

def load_embedding_model(model_type, embedding_dimension=50):
    """
    Loads a pre-trained word embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """

    download_path = ""

    # Find the correct embedding model name
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"

    elif model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)

    else:
        raise AttributeError("Unsupported embedding model type! Available ones: word2vec, glove")

    # Check download
    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Word2Vec: 300")
        print("Glove: 50, 100, 200, 300")
        raise e

    return emb_model


# Modify these variables as you wish!
# Glove -> 50, 100, 200, 300
embedding_model_type = "glove"
embedding_dimension = 300

embedding_model = load_embedding_model(embedding_model_type, embedding_dimension)

"""## Dataset Parameters

"""

sents_len = []
for x in X_train_encoded:
  sents_len.append(len(x))

MAX_SEQ_LENGTH = 41  # median of lenghts
AVG_SEQ_LENGHT = int(np.mean(sents_len))  # mean of lenghts

MAX_SEQ_LENGTH = AVG_SEQ_LENGHT

"""## Create dataset"""

#@title
'''
from math import floor, ceil

def normalize_lenght_dataset(X_train, Y_train):
  x_dataset = []
  y_dataset = []
  for (x, y) in zip(X_train, Y_train):
      sent_len = len(x)
      if sent_len > MAX_SEQ_LENGTH:
        for i in range(round(sent_len/MAX_SEQ_LENGTH)):
          x_temp = np.zeros(MAX_SEQ_LENGTH, dtype=np.int64)
          x_remain = x[MAX_SEQ_LENGTH*(i): (MAX_SEQ_LENGTH*(i+1) if sent_len > MAX_SEQ_LENGTH*(i+1) else len(x))] 
          x_temp[:MAX_SEQ_LENGTH if sent_len > MAX_SEQ_LENGTH*(i+1) else len(x)-MAX_SEQ_LENGTH*(i)] = x_remain
          x_dataset.append(x_temp)

          y_dataset.append(y[MAX_SEQ_LENGTH*(i): (MAX_SEQ_LENGTH*(i+1))])
      else:
        x_temp = np.zeros(MAX_SEQ_LENGTH, dtype=np.int64)
        x_temp[:len(x)] = x
        x_dataset.append(x_temp)
        y_dataset.append(y[:MAX_SEQ_LENGTH])

  return tf.constant(x_dataset), tf.constant(y_dataset)


# faccio il padding perchè è un tensore 3d ed è più conveniente 
Y_train = pad_sequences(Y_train, maxlen=300, padding="post", truncating="post")
Y_validate = pad_sequences(Y_validate, maxlen=300, padding="post", truncating="post")
Y_test = pad_sequences(Y_test, maxlen=300, padding="post", truncating="post")
X_train, Y_train = normalize_lenght_dataset(X_train_encoded, Y_train)
X_validate, Y_validate = normalize_lenght_dataset(X_validate_encoded, Y_validate)
X_test, Y_test = normalize_lenght_dataset(X_test_encoded, Y_test)


print("len(X_train) {}, len(Y_train) {}".format(X_train.shape, Y_train.shape ))
print("len(X_validate) {}, len(Y_validate) {}".format(X_validate.shape, Y_validate.shape ))
print("len(X_test) {}, len(Y_test) {}".format(X_test.shape, Y_test.shape ))
'''

X_train = pad_sequences(X_train_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
Y_train = pad_sequences(Y_train, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")

X_validate = pad_sequences(X_validate_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
Y_validate = pad_sequences(Y_validate, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")

X_test = pad_sequences(X_test_encoded, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")
Y_test = pad_sequences(Y_test, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post")

print("len(X_train) {}, len(Y_train) {}".format(X_train.shape, Y_train.shape ))
print("len(X_validate) {}, len(Y_validate) {}".format(X_validate.shape, Y_validate.shape ))
print("len(X_test) {}, len(Y_test) {}".format(X_test.shape, Y_test.shape ))

# Function definition

def build_embedding_matrix(embedding_model, word_list):
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_list: vocabulary map (word -> index) (dict)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """

    ### YOUR CODE HERE ###
    V = len(word_list)
    D = embedding_dimension
    embedding_matrix = np.zeros(shape=(V,D)) 
    for i in range(len(word_list)):
      curr_word = word_list[i]
      if curr_word in embedding_model.vocab:
        embedding_matrix[i] = embedding_model[curr_word]
      else:
        embedding_matrix[i] = np.zeros(shape=D)
    for j in range(V):
      if not (embedding_matrix[j].any()):
        if j != 0 and j < V-1:
          embedding_matrix[j] = np.mean([embedding_matrix[j-1],embedding_matrix[j+1]])
        if j==V-1:
          embedding_matrix[j] = np.mean([embedding_matrix[j-1],embedding_matrix[0]])


    return embedding_matrix

embedding_matrix_train_validate = build_embedding_matrix(embedding_model, train_val_vocab)
embedding_matrix_test = build_embedding_matrix(embedding_model, test_vocab)

print("Embedding matrix train_val shape: {}".format(embedding_matrix_train_validate.shape))
print("Embedding matrix test shape: {}".format(embedding_matrix_test.shape))

"""# Models Parameters"""

baseline_rnn_units = 512
baseline_epochs = 50
baseline_batch_size = 20

double_lstm_rnn_units = 512
double_lstm_epochs = 50
double_lstm_batch_size = 20

gru_rnn_units = 512
gru_epochs = 50
gru_batch_size = 20

crf_rnn_units = 512
crf_epochs = 50
crf_batch_size = 20

#Prepare the report file

import os
if os.path.exists("Scores.txt"):
  print("Deleting old scores file:\n")
  os.remove("Scores.txt")


f = open("Scores.txt", "x")

"""# BASELINE LSTM"""

def flatten_y(Y_train):
  Y_train_crf = []
  for i in range(len(Y_train)):
    Y_train_crf.append([])
    for j in range(len(Y_train[0])):
      Y_train_crf[i].append(np.argmax(list(Y_train[i][j])))

  return tf.constant(Y_train_crf)

"""## Callback"""

callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val-accuracy', patience=5)

def exp_decay_lr(epoch):
   initial_lrate = 0.01
   k = 0.1
   lrate = initial_lrate * math.exp(-k*epoch)
   return lrate

callback_adaptive_lr = tf.keras.callbacks.LearningRateScheduler(exp_decay_lr)

"""## Model"""

model = Sequential()
model.add(Input((MAX_SEQ_LENGTH,)))
model.add(Masking(mask_value=0.0))
model.add(Embedding(len(train_val_vocab),
                embedding_dimension,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix_train_validate),
                trainable=False))
model.add(Bidirectional(LSTM(baseline_rnn_units, return_sequences=True)))
model.add(TimeDistributed(Dense(len(pos_tags), activation='softmax')))
model.summary()

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

"""## Fit"""

model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), batch_size=baseline_batch_size, epochs=baseline_epochs, callbacks=[callback_early_stopping, callback_adaptive_lr])
print("Training ended")

"""## Evaluation"""

score, acc = model.evaluate(X_validate, Y_validate, batch_size=baseline_batch_size)
print('Score Baseline:', score)
print('Accuracy Baseline:', acc)
Y_validate_predicted = model.predict(X_validate)
print(flatten_y(Y_validate_predicted))
f1_score_macro = f1_score(
    tf.keras.backend.flatten(flatten_y(Y_validate)), 
    tf.keras.backend.flatten(flatten_y(Y_validate_predicted)), 
    average='macro')
print("F1 Macro Baseline: {}".format(f1_score_macro))

f.write("Baseline Model: \n")
scores = ["Score: {}\n".format(score), "Accuracy: {}\n".format(acc), "F1 Score: {}".format(f1_score_macro), "\n\n"]
f.write(scores)

"""# GRU

## Model
"""

model_GRU = Sequential()
model_GRU.add(Input((MAX_SEQ_LENGTH,)))
model_GRU.add(Embedding(len(train_val_vocab),
                embedding_dimension,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix_train_validate),
                trainable=False,
                mask_zero=True))
model_GRU.add(Bidirectional(GRU(gru_rnn_units, return_sequences=True)))
model_GRU.add(TimeDistributed(Dense(len(pos_tags), activation='softmax')))
model_GRU.summary()

model_GRU.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

"""## Fit"""

model_GRU.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), batch_size=gru_batch_size, epochs=gru_epochs, callbacks=[callback_early_stopping, callback_adaptive_lr])
print("Training ended")

"""## Evaluation"""

score, acc = model_GRU.evaluate(X_validate, Y_validate, batch_size=gru_batch_size)
print('Score GRU:', score)
print('Accuracy GRU:', acc)
Y_validate_predicted = model_GRU.predict(X_validate)
f1_score_macro = f1_score(
    tf.keras.backend.flatten(flatten_y(Y_validate)), 
    tf.keras.backend.flatten(flatten_y(Y_validate_predicted)), 
    average='macro')
print("F1 Macro GRU: {}".format(f1_score_macro))

f.write("GRU Model: \n")
scores = ["Score: {}\n".format(score), "Accuracy: {}\n".format(acc), "F1 Score: {}".format(f1_score_macro), "\n\n"]
f.write(scores)

"""# LSTM DOUBLE

## Model
"""

model_double_LSTM = Sequential()
model_double_LSTM.add(Input((MAX_SEQ_LENGTH,)))
model_double_LSTM.add(Embedding(len(train_val_vocab),
                embedding_dimension,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix_train_validate),
                trainable=False,
                mask_zero=True))
model_double_LSTM.add(Bidirectional(LSTM(double_lstm_rnn_units, return_sequences=True)))
model_double_LSTM.add(LSTM(double_lstm_rnn_units, return_sequences=True))
model_double_LSTM.add(TimeDistributed(Dense(len(pos_tags), activation='softmax')))
model_double_LSTM.summary()

model_double_LSTM.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

"""## Fit"""

model_double_LSTM.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), batch_size=double_lstm_batch_size, epochs=double_lstm_epochs, callbacks=[callback_early_stopping, callback_adaptive_lr])
print("Training ended")

"""## Evaluation"""

score, acc = model_double_LSTM.evaluate(X_validate, Y_validate, batch_size=double_lstm_batch_size)
print('Score Double LSTM:', score)
print('Accuracy Double LSTM:', acc)
Y_validate_predicted = model_double_LSTM.predict(X_validate)
f1_score_macro = f1_score(
    tf.keras.backend.flatten(flatten_y(Y_validate)), 
    tf.keras.backend.flatten(flatten_y(Y_validate_predicted)), 
    average='macro')
print("F1 Macro Double LSTM: {}".format(f1_score_macro))

f.write("Double LSTM Model: \n")
scores = ["Score: {}\n".format(score), "Accuracy: {}\n".format(acc), "F1 Score: {}".format(f1_score_macro), "\n\n"]
f.write(scores)

"""# CRF

## Model
"""

inputs = Input((MAX_SEQ_LENGTH, ))
output = Embedding(len(train_val_vocab),
                embedding_dimension,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix_train_validate),
                trainable=False,
                mask_zero=True)(inputs)
output = Bidirectional(LSTM(crf_rnn_units, return_sequences=True))(output)
output = TimeDistributed(Dense(len(pos_tags)))(output)

crf = CRF(dtype='float32')
output = crf(output)
base_model = Model(inputs, output)
model_CRF = ModelWithCRFLoss(base_model)
model_CRF.compile(optimizer='adam')
model_CRF.build((None, MAX_SEQ_LENGTH))
model_CRF.summary()

"""## Fit"""

Y_train_crf = flatten_y(Y_train)
Y_validate_crf = flatten_y(Y_validate)
model_CRF.fit(X_train, Y_train_crf, validation_data=(X_validate, Y_validate_crf), batch_size=crf_batch_size, epochs=gru_epochs, callbacks=[callback_early_stopping, callback_adaptive_lr])
print("Training ended")

"""## Evaluation"""

Y_validate_predicted = model_CRF.predict(X_validate)
Y_validate_predicted = Y_validate_predicted[0]
m = tf.keras.metrics.Accuracy()
m.update_state(Y_validate_crf, Y_validate_predicted)
print("Accuracy CRF: {}".format(m.result().numpy()))

f1_score_macro = f1_score(
    tf.keras.backend.flatten(Y_validate_crf), 
    tf.keras.backend.flatten(Y_validate_predicted), 
    average='macro')
print("F1 Macro CRF: {}".format(f1_score_macro))

f.write("CRF Model: \n")
scores = ["Score: {}\n".format(score), "Accuracy: {}\n".format(acc), "F1 Score: {}".format(f1_score_macro), "\n\n"]
f.write(scores)


f.close()

#Saving full report
from google.colab import files
files.download('Scores.txt')

"""# BEST MODEL"""

# BEST MODEL 
best_model = model  # FILL WITH THE BEST MODEL
is_CRF_the_best = False  # Change to True if the best model is CRF, because it's different data it needs different evaluation

def remove_padding(y_true, y_pred):
  no_padding_vector_true = []  #np.zeros((BATCH_SIZE, y_true.shape[1], y_true.shape[2]))
  no_padding_vector_pred = []  #np.zeros((BATCH_SIZE, y_true.shape[1], y_true.shape[2]))
  for i in range(BATCH_SIZE): # per ogni frase inserisco una lista
    no_padding_vector_true.append([])
    no_padding_vector_pred.append([])  # vettore per la frase no_padding_vector_pred[i]
    
    for j in range(y_true.shape[1]):
      if sum(y_true[i][j].numpy()) != 0.:  # per ogni parola non padding la aggiungo
        assert len(list(y_true[i][j].numpy())) == len(list(y_pred[i][j]))
        no_padding_vector_true[i].append(np.argmax(list(y_true[i][j].numpy())))
        no_padding_vector_pred[i].append(np.argmax(list(y_pred[i][j])))

  return no_padding_vector_true, no_padding_vector_pred

if is_CRF_the_best:

  Y_test_predicted = best_model.predict(X_test)
  Y_test_no_padding, Y_test_predicted_no_padding = remove_padding(Y_test, Y_test_predicted)

  score, acc = model_CRF.evaluate(X_test, Y_test, batch_size=crf_batch_size)
  print('Test score:', score)
  print('Test accuracy:', acc)

  print("#### REPORT ####")

  Y_test_no_padding = [item for sublist in Y_test_no_padding for item in sublist]
  Y_test_predicted_no_padding = [item for sublist in Y_test_predicted_no_padding for item in sublist]

  print(classification_report(Y_test_no_padding, Y_test_predicted_no_padding))
else:
  score, acc = model_GRU.evaluate(X_test, Y_test, batch_size=crf_batch_size)
  print('Test score:', score)
  print('Test accuracy:', acc)
  Y_test_predicted = model_GRU.predict(X_test)
  print(f1_score(
      tf.keras.backend.flatten(flatten_y(Y_test)), 
      tf.keras.backend.flatten(flatten_y(Y_test_predicted)), 
      average='macro'))