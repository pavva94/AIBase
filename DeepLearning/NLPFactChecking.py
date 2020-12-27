# -*- coding: utf-8 -*-
"""
**Summary**: Fact checking, Neural Languange Inference (**NLI**)

Steps:
*   Dataset preparation (analysis and pre-processing)
*   Problem formulation: multi-input binary classification
*   Defining an evaluation method
*   Simple sentence embedding
*   Neural building blocks
*   Neural architecture extension

"""

import os
import requests
import zipfile

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_data(data_path):
    toy_data_path = os.path.join(data_path, 'fever_data.zip')
    toy_data_url_id = "1wArZhF9_SHW17WKNGeLmX-QTYw9Zscl1"
    toy_url = "https://docs.google.com/uc?export=download"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(toy_data_path):
        print("Downloading FEVER data splits...")
        with requests.Session() as current_session:
            response = current_session.get(toy_url,
                                   params={'id': toy_data_url_id},
                                   stream=True)
        save_response_content(response, toy_data_path)
        print("Download completed!")

        print("Extracting dataset...")
        with zipfile.ZipFile(toy_data_path) as loaded_zip:
            loaded_zip.extractall(data_path)
        print("Extraction completed!")

download_data('dataset')

## Imports

import re
import datetime
import math

import pandas as pd
import numpy as np
import gensim
import gensim.downloader as gloader

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Masking, GRU, Dropout, Dot, Concatenate, Add, Average
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from unidecode import unidecode

from scipy.spatial.distance import cosine as cosine_similarity


"""## Dataset pre-processing

### Read csv
"""

train_dataframe = pd.read_csv('dataset/train_pairs.csv', index_col=0)  
val_dataframe = pd.read_csv('dataset/val_pairs.csv', index_col=0)  
test_dataframe = pd.read_csv('dataset/val_pairs.csv', index_col=0)

"""### Embedding model"""

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
    if model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)

    else:
        raise AttributeError("Unsupported embedding model type! Available ones: word2vec, glove")

    # Check download
    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name! Check the embedding dimension:")
        print("Glove: 50, 100, 200, 300")
        raise e

    return emb_model

# Function definition

def build_embedding_matrix(embedding_model, word_list):
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_list: vocabulary map (word -> index) (dict)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """
    V = len(word_list)
    D = embedding_dimension
    embedding_matrix = np.zeros(shape=(V,D)) 
    for i in range(len(word_list)):
      curr_word = word_list[i]
      if curr_word in embedding_model.vocab:
        embedding_matrix[i] = embedding_model[curr_word].tolist()
      else:
        embedding_matrix[i] = np.zeros(shape=D)
    for j in range(V):
      if not (embedding_matrix[j].any()):
        if j != 0 and j < V-1:
          embedding_matrix[j] = np.mean([embedding_matrix[j-1],embedding_matrix[j+1]], axis=0).tolist()
        if j==V-1:
          embedding_matrix[j] = np.mean([embedding_matrix[j-1],embedding_matrix[0]], axis=0).tolist()

    return embedding_matrix

def remove_non_ascii(text):
  encoded_string = text.encode("ascii", "ignore")
  return encoded_string.decode()

def create_dataset_vocabulary(df):
  # create the dataset and the vocabulary from the raw dataframe
  dataset, vocabulary = [], []
  claim_length = []
  evidence_length = []

  for index, row in df.iterrows():
    claim = re.sub(r'[^a-zA-Z0-9 ]+', "", remove_non_ascii(row['Claim']))
    evidences_init = remove_non_ascii(row['Evidence']).replace("...", "")
    evidences_init = evidences_init.split('.')[0].split('\t')[1]  # extract the evidence before the first '.' and remove the number before the first '\t'
    evidence = re.sub(r'[^a-zA-Z0-9 ]+', "", evidences_init)

    label = row['Label']

    # add words to vocabulary
    claim_word_list = [a for a in claim.split(" ") if a is not ""]
    vocabulary.extend(claim_word_list)
    claim_length.append(len(claim_word_list))

    evidence_word_list = [a for a in evidence.split(" ") if a.strip() is not ""]
    vocabulary.extend(evidence_word_list)
    
    evidence_length.append(len(evidence_word_list))
    dataset.append((claim, evidence, label))  # we have the same label for every evidence

  #Compute AVG_LENGTH of claims and evidences
  AVG_CLAIM_LEN = sum(claim_length)/len(claim_length)
  AVG_EVIDENCE_LEN = sum(evidence_length)/len(evidence_length)

  final_dataset = pd.DataFrame(dataset, columns=['claim', 'evidence', 'label'])
  final_dataset = final_dataset.drop_duplicates()
  return final_dataset, set(vocabulary), AVG_CLAIM_LEN, AVG_EVIDENCE_LEN

from itertools import chain, repeat, islice

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

"""### Creation of the encoded dataset"""

#Creation of the encoded dataset, for the bag of vectors case the flag bag_of_vectors must be true
def create_encoded_dataset(df, w2i, embedding_matrix, label_encoding, AVG_CLAIM_LEN, AVG_EVIDENCE_LEN, embedding_dim, bag_of_vectors=False):
  # here we have the dataset [batch_size, max_tokens] for each claim and evidence
  x_dataset = []
  y_dataset = []

  for index, row in df.iterrows():
    claim = row['claim']
    evidence = row['evidence']
    # encode the label
    y_dataset.append(label_encoding[row['label']])

    claim_encoded = []
    # encode the claim with the embedding [batch_size, max_tokens, embedding_dim]
    # embed each word of the claim
    for c in claim.strip().split(" "):
      if c.strip() is not "":
        claim_encoded.append(embedding_matrix[w2i[c.strip()]])

    # reduce the claim to [batch_size, embedding_dim]
    if bag_of_vectors:
      if len(claim_encoded) < 2:
        print("index {}, len {}".format(index, len(claim_encoded)))
      claim_encoded = np.mean(claim_encoded, axis=0)
    
    # same procedure for evidence
    # embed each word of the evidence
    evidence_encoded = []
    for ev in evidence.strip().split(" "):
      if ev.strip() is not "":
        evidence_encoded.append(embedding_matrix[w2i[ev.strip()]])
    if bag_of_vectors:
      evidence_encoded = np.mean(evidence_encoded, axis=0)

    if not bag_of_vectors:
      claim_encoded = list(pad(claim_encoded, AVG_CLAIM_LEN, padding=np.zeros(embedding_dim)))
      evidence_encoded = list(pad(evidence_encoded, AVG_EVIDENCE_LEN, padding=np.zeros(embedding_dim)))

    # create the final dataset [batch_size, (claim, evidence), max_tokens, embedding_dim]
    x_dataset.append([claim_encoded, evidence_encoded])  # we have the same label for every claim

  return x_dataset, y_dataset

"""### Preparing the embedding matrix and dataframe


"""

# Glove -> 50, 100, 200, 300
embedding_model_type = "glove"
embedding_dimension = 200
#if not embedding_model:
embedding_model = load_embedding_model(embedding_model_type, embedding_dimension)

# [batch_size, max_tokens], set(str)
train_df, train_vocab, train_claim_len, train_evidence_len = create_dataset_vocabulary(train_dataframe)
val_df, val_vocab, val_claim_len, val_evidence_len = create_dataset_vocabulary(val_dataframe)
test_df, test_vocab, test_claim_len, test_evidence_len = create_dataset_vocabulary(test_dataframe)

AVG_CLAIM_LEN = int((train_claim_len + val_claim_len + test_claim_len) / 3)
AVG_EVIDENCE_LEN = int((train_evidence_len + val_evidence_len + test_evidence_len) / 3)

train_val_vocab = set(list(train_vocab) + list(val_vocab))

embedding_matrix_train_validate = build_embedding_matrix(embedding_model, list(train_val_vocab))
embedding_matrix_test = build_embedding_matrix(embedding_model, list(test_vocab))

print("Train dataframe head {}".format(train_df.head()))
print("Train vocab length {}".format(len(train_vocab)))
print("Validation vocab length {}".format(len(val_vocab)))
print("Test vocab length {}".format(len(test_vocab)))
print("Embedding matrix train_val shape: {}".format(embedding_matrix_train_validate.shape))
print("Embedding matrix test shape: {}".format(embedding_matrix_test.shape))

train_df.iloc[2908]

# Store vocab in a dict, associated with a numerical index and creation of the label encoding
train_val_word2idx = { word[1]: word[0] for word in enumerate(train_val_vocab) }
train_val_idx2word = { v: k for k, v in train_val_word2idx.items() }

test_word2idx = { word[1]: word[0] for word in enumerate(test_vocab) }
test_idx2word = { v: k for k, v in test_word2idx.items() }

label_encoding = {'SUPPORTS': 1, 'REFUTES': 0}

"""## Merging layer """

class MergingLayer(tf.keras.layers.Layer):
  def __init__(self, merging_type, cosine_sim_flag=False):
    super(MergingLayer, self).__init__()
    self.merging_type = merging_type
    self.cosine_sim_flag = cosine_sim_flag

  def call(self, claim, evidence):
    
    if self.merging_type == 'concat':
      new_row = Concatenate(axis=1)([claim, evidence])
    elif self.merging_type == 'sum':
      new_row = Add()([claim, evidence])
    elif self.merging_type == 'mean':
      new_row = Average()([claim, evidence])

    if self.cosine_sim_flag:
      cosine_sim = Dot(1, normalize=True)([claim, evidence])
      new_row = Concatenate(axis=1)([new_row, cosine_sim])
    return new_row

"""## Each model with same merging strategy

1. RNN First encode token sequences and take the last state as the sentence embedding.
2. RNN Second encode token sequences and average all the output states.
3. MLP for encoding token sequences.

These previous points are developed separately as a single NN that encode and classify.

4. BoV + MLP

#### Callbacks
"""

callback_early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
def exp_decay_lr(epoch):
   initial_lrate = 0.01
   k = 0.1
   lrate = initial_lrate * math.exp(-k*epoch)
   return lrate

callback_adaptive_lr = LearningRateScheduler(exp_decay_lr)

"""### Encoding of the Dataset"""

#Creation of the dataset for the RNN cases
#Transforming train, val, test word lists in id lists [batch_size, embedding_dim]
X_train_encoded, Y_train_encoded = create_encoded_dataset(train_df, train_val_word2idx, embedding_matrix_train_validate, label_encoding, AVG_CLAIM_LEN, AVG_CLAIM_LEN, embedding_dimension)
X_validate_encoded, Y_validate_encoded = create_encoded_dataset(val_df, train_val_word2idx, embedding_matrix_train_validate, label_encoding, AVG_CLAIM_LEN, AVG_CLAIM_LEN, embedding_dimension)
X_test_encoded, Y_test_encoded = create_encoded_dataset(test_df, test_word2idx, embedding_matrix_test, label_encoding, AVG_CLAIM_LEN, AVG_CLAIM_LEN, embedding_dimension)

# convert to tensor
X_train_encoded, Y_train_encoded = tf.constant(X_train_encoded, dtype=tf.float32), tf.constant(Y_train_encoded, dtype=tf.float32)
X_validate_encoded, Y_validate_encoded = tf.constant(X_validate_encoded, dtype=tf.float32), tf.constant(Y_validate_encoded, dtype=tf.float32)
X_test_encoded, Y_test_encoded = tf.constant(X_test_encoded, dtype=tf.float32), tf.constant(Y_test_encoded, dtype=tf.float32)

print(X_train_encoded.shape)

"""### RNN First case"""

rnn_units_claim = embedding_dimension
rnn_batch_size = 64
rnn_epochs = 50
rnn_unit_dense_1 = 150
rnn_unit_dense_2 = 100

log_dir = "logs/fit/RNNFirst/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_rnn = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class RNNFirst(tf.keras.Model):
  def __init__(self, enc_units, unit_dense_1, unit_dense_2, merging_type='concat', cosine_sim=False):
    super(RNNFirst, self).__init__()
    self.enc_units = enc_units
    self.rnn_1 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    self.rnn_2 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    
    self.merging_layer = MergingLayer(merging_type, cosine_sim)
    
    self.dense_1 = Dense(unit_dense_1, kernel_regularizer='l2')
    self.dense_2 = Dense(unit_dense_2, kernel_regularizer='l2')
    self.out = Dense(1, activation = 'sigmoid')

  def call(self, x_input):

    c, e = x_input[:, 0, :, :], x_input[:, 1, :, :]   # flatten_dataset(tf.make_ndarray(tf.make_tensor_proto(x_input))

    _, encoded_state_claim, _ = self.rnn_1(c)
    del c
    
    _, encoded_state_evidence, _ = self.rnn_2(e)
    del e
    
    x = self.merging_layer(encoded_state_claim, encoded_state_evidence)
    
    x = self.dense_1(x)
    x = Dropout(0.3)(x)
    x = self.dense_2(x)
    x = Dropout(0.3)(x)

    output = self.out(x)

    
    return output


rnn_first = RNNFirst(embedding_dimension, rnn_unit_dense_1, rnn_unit_dense_2, merging_type='concat')
rnn_first.compile(optimizer='nadam',  # GRU adam:valaccuracy 71.4 vallos 98.9, sgd:valaccuracy 65.6 vallos 66.0, Nadam:valaccuracy 72.0 vallos 104.9, 
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_first.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn_batch_size,
          epochs=rnn_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn])

loss, accuracy = rnn_first.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # concat 72.1 sum 72.1 mean 72.5

"""### RNN Second case"""

rnn2_units_claim = embedding_dimension
rnn2_batch_size = 64
rnn2_epochs = 50
rnn2_unit_dense_1 = 100
rnn2_unit_dense_2 = 50

log_dir = "logs/fit/RNNSecond/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_rnn2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class RNNSecond(tf.keras.Model):
  def __init__(self, enc_units, unit_dense_1, unit_dense_2, merging_type='concat', cosine_sim=False):
    super(RNNSecond, self).__init__()
    self.enc_units = enc_units
    self.rnn_1 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    self.rnn_2 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    
    self.merging_layer = MergingLayer(merging_type, cosine_sim)
    
    self.dense_1 = Dense(unit_dense_1, kernel_regularizer='l2')
    self.dense_2 = Dense(unit_dense_2, kernel_regularizer='l2')
    self.out = Dense(1, activation = 'sigmoid')

  def call(self, x_input):

    c, e = x_input[:, 0, :, :], x_input[:, 1, :, :] # flatten_dataset(tf.make_ndarray(tf.make_tensor_proto(x_input))

    output_vector_claim, _, _ = self.rnn_1(c)
    del c
    
    output_vector_evidence, _, _ = self.rnn_2(e)  # [batch_size, max_length, embedding_dim]
    del e

    claim = tf.math.reduce_mean(output_vector_claim, axis=1)
    evidence = tf.math.reduce_mean(output_vector_evidence, axis=1)  # [batch_size, embedding_dim]
    
    x = self.merging_layer(claim, evidence)
    
    x = self.dense_1(x)
    x = Dropout(0.3)(x)
    x = self.dense_2(x)
    x = Dropout(0.3)(x)
    output = self.out(x)

    
    return output


rnn_second = RNNSecond(embedding_dimension, rnn2_unit_dense_1, rnn2_unit_dense_2, merging_type='concat')
rnn_second.compile(optimizer='nadam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_second.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn2_batch_size,
          epochs=rnn2_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn2])  # 74.95 max val acc

loss, accuracy = rnn_second.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # concat 71.5 sum 71.7 mean

"""### MLP"""

mlp_units_claim = embedding_dimension
mlp_batch_size = 64
mlp_epochs = 50
mlp_unit_dense_1 = 50 
mlp_unit_dense_2 = 25

log_dir = "logs/fit/MLP/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_mlp = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class MLP(tf.keras.Model):
  def __init__(self, enc_units, unit_dense_1, unit_dense_2, merging_type='concat', cosine_sim=False):
    super(MLP, self).__init__()
    self.encoder_claim = Dense(enc_units, trainable=False)
    self.encoder_evidence = Dense(enc_units, trainable=False)
    
    self.merging_layer = MergingLayer(merging_type, cosine_sim)
    
    self.dense_1 = Dense(unit_dense_1, kernel_regularizer='l2')
    self.dense_2 = Dense(unit_dense_2, kernel_regularizer='l2')
    self.out = Dense(1, activation = 'sigmoid')

  def call(self, x_input):

    c, e = x_input[:, 0, :, :], x_input[:, 1, :, :] # flatten_dataset(tf.make_ndarray(tf.make_tensor_proto(x_input))

    output_vector_claim = self.encoder_claim(c)
    del c
    
    output_vector_evidence = self.encoder_evidence(e)  # [batch_size, embedding_dim]
    del e
    
    x = self.merging_layer(output_vector_claim, output_vector_evidence)
    
    x = self.dense_1(x)

    x = self.dense_2(x)

    output = self.out(x)

    return output


mlp = MLP(embedding_dimension, mlp_unit_dense_1, mlp_unit_dense_2, merging_type='concat')
mlp.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

mlp.fit(X_train_encoded, Y_train_encoded, 
          batch_size=mlp_batch_size,
          epochs=mlp_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_mlp])

loss, accuracy = mlp.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # 51.3

"""### RNN Encoder without training

A simple RNN that encode the input.
"""

def flatten_dataset(X_ds):
  claim_list = []
  evidence_list = []
  for (c,e) in X_ds:
    claim_list.append(c)
    evidence_list.append(e)
  return claim_list, evidence_list

train_claim_list, train_evidence_list = flatten_dataset(X_train_encoded)
val_claim_list, val_evidence_list = flatten_dataset(X_validate_encoded)
test_claim_list, test_evidence_list = flatten_dataset(X_test_encoded)

class Encoder(tf.keras.Model):
  def __init__(self, max_length, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)

  def call(self, x):
    
    output, state = self.gru(x)
    return output, state

claim_encoder = Encoder(AVG_CLAIM_LEN, embedding_dimension, rnn_units_claim)

train_claim_output, train_claim_hidden = claim_encoder.predict(train_claim_list)
val_claim_output, val_claim_hidden = claim_encoder.predict(val_claim_list)
test_claim_output, test_claim_hidden = claim_encoder.predict(test_claim_list)

evidence_encoder = Encoder(AVG_EVIDENCE_LEN, embedding_dimension, rnn_units_claim)

train_evidence_output, train_evidencem_hidden = evidence_encoder.predict(train_evidence_list)
val_evidence_output, val_evidence_hidden = evidence_encoder.predict(val_evidence_list)
test_evidence_output, test_evidence_hidden = evidence_encoder.predict(test_evidence_list)

"""### Bag Of Vectors"""

#Creation of the dataset for the bag of vectors case
#Transforming train, val, test word lists in id lists [batch_size, embedding_dim]
X_train_bov, Y_train_bov = create_encoded_dataset(train_df, train_val_word2idx, embedding_matrix_train_validate, label_encoding, AVG_CLAIM_LEN, AVG_CLAIM_LEN, embedding_dimension, bag_of_vectors=True)
X_validate_bov, Y_validate_bov = create_encoded_dataset(val_df, train_val_word2idx, embedding_matrix_train_validate, label_encoding, AVG_CLAIM_LEN, AVG_CLAIM_LEN, embedding_dimension, bag_of_vectors=True)

def concatenate_claim_evidence(X_train):
  X_concatenated = []
  for x in X_train:
    cosine_sim = cosine_similarity(x[0], x[1])
    new_row = np.concatenate((x[0], x[1])).tolist() + [cosine_sim]
    X_concatenated.append(new_row)
  return X_concatenated

def sum_claim_evidence(X_train):
  X_sum = []
  for x in X_train:
    cosine_sim = cosine_similarity(x[0], x[1])
    new_row = np.add(x[0], x[1]).tolist() + [cosine_sim]
    X_sum.append(new_row)
  return X_sum

def mean_claim_evidence(X_train):
  X_mean = []
  for x in X_train:
    cosine_sim = cosine_similarity(x[0], x[1])
    new_row = np.mean((x[0], x[1]), axis=0).tolist() + [cosine_sim]
    X_mean.append(new_row)
  return X_mean

# Concatenation [batch_size, 2 * embedding_dim + 1]
X_train_concatenated = concatenate_claim_evidence(X_train_bov)
X_validate_concatenated = concatenate_claim_evidence(X_validate_bov)
X_test_concatenated = concatenate_claim_evidence(X_test_bov)

# Sum [batch_size, embedding_dim + 1]
# X_train_summed = sum_claim_evidence(X_train_bov)
# X_validate_summed = sum_claim_evidence(X_validate_bov)
# X_test_summed = sum_claim_evidence(X_test_bov)

# Mean [batch_size, embedding_dim + 1]
# X_train_mean = mean_claim_evidence(X_train_bov)
# X_validate_mean = mean_claim_evidence(X_validate_bov)
# X_test_mean = mean_claim_evidence(X_test_bov)

"""#### MLP

"""

log_dir = "logs/fit/BoV/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_bov = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

unit_dense_1 = 200
unit_dense_2 = 100
num_batches = 32
num_epochs = 100

model = Sequential([
  Input((401, )),
  Dense(unit_dense_1, kernel_regularizer='l2'),
  Dense(unit_dense_2, kernel_regularizer='l2'),
  Dense(1, activation = 'sigmoid')])

model.build((None, 101))
model.summary()

model.compile(optimizer='nadam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train_concatenated,
                    Y_train_encoded,
                    batch_size=num_batches,
                    epochs=num_epochs,
                    validation_data=(X_validate_concatenated, Y_validate_encoded),
                    validation_steps=30,
                    callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_bov])

loss, accuracy = model.evaluate(X_validate_concatenated, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # 50.4

"""### Check best model with Test Dataset"""

loss, accuracy = rnn_first.evaluate(X_test_encoded, Y_test_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

"""## Best model with different merging strategy"""

rnn_best_units_claim = embedding_dimension
rnn_best_batch_size = 64
rnn_best_epochs = 50
rnn_best_unit_dense_1 = 150
rnn_best_unit_dense_2 = 100

"""### Sum"""

rnn_best_sum = RNNFirst(rnn_best_units_claim, rnn_best_unit_dense_1, rnn_best_unit_dense_2, merging_type='sum')
rnn_best_sum.compile(optimizer='nadam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_best_sum.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn_best_batch_size,
          epochs=rnn_best_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn])

loss, accuracy = rnn_best_sum.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # 72.1

"""### Mean"""

rnn_best_mean = RNNFirst(embedding_dimension, rnn_best_unit_dense_1, rnn_best_unit_dense_2, merging_type='mean')
rnn_best_mean.compile(optimizer='nadam', 
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_best_mean.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn_best_batch_size,
          epochs=rnn_best_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn])

loss, accuracy = rnn_best_mean.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # 72.5

"""### Check best model with best merging strategy on Test Dataset"""

loss, accuracy = rnn_best_sum.evaluate(X_test_encoded, Y_test_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

"""## Best model with best merging strategy and cosine similarity"""

rnn_best_cosine = RNNFirst(embedding_dimension, rnn_unit_dense_1, rnn_unit_dense_2, merging_type='mean', cosine_sim=True)
rnn_best_cosine.compile(optimizer='nadam', 
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_best_cosine.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn_batch_size,
          epochs=rnn_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn])

loss, accuracy = rnn_best_cosine.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

"""### Check best model with best merging srategy and cosine simliarity on Test Dataset"""

loss, accuracy = rnn_best_cosine.evaluate(X_test_encoded, Y_test_encoded)

print("Test Loss: ", loss)
print("Test Accuracy: ", accuracy)

"""## Extras

### RNNFirst with Attention
"""

class AttentionRNN(tf.keras.layers.Layer):
  def __init__(self, units):
    super(AttentionRNN, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):  #(self, hidden_state_rnn, output_rnn)
    query_with_time_axis = tf.expand_dims(query, 1)  # output: (None, 1, 200)

    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))  # output: (None, 8, 1)

    attention_weights = tf.nn.softmax(score, axis=1)  # output: (None, 8, 1)

    context_vector = attention_weights * values  # (None, 8, 1) * (None, 8, 200)
    context_vector = tf.reduce_sum(context_vector, axis=1)  # (None, 200)
    
    return context_vector, attention_weights

# Creare una rete con due lstm paralleli, uno per claim uno per evidence, 
# concatenare hidden states delle due lstm + aggiungere cosine similarity, 
# poi passare tutto all' MLP
rnn_attention_units_claim = embedding_dimension
rnn_attention_batch_size = 64
rnn_attention_epochs = 50
rnn_attention_unit_dense_1 = 100
rnn_attention_unit_dense_2 = 50

log_dir = "logs/fit/RNNAttention/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_rnn_attention = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class RNNFirstAttention(tf.keras.Model):
  def __init__(self, enc_units, unit_dense_1, unit_dense_2, merging_type='concat'):
    super(RNNFirstAttention, self).__init__()
    self.enc_units = enc_units
    self.rnn_1 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    self.rnn_2 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    
    self.merging_layer = MergingLayer(merging_type)

    self.attention_claim = AttentionRNN(self.enc_units)
    self.attention_evidence = AttentionRNN(self.enc_units)
    
    self.dense_1 = Dense(unit_dense_1, kernel_regularizer='l2')
    self.dense_2 = Dense(unit_dense_2, kernel_regularizer='l2')
    self.out = Dense(1, activation = 'sigmoid')

  def call(self, x_input):

    c, e = x_input[:, 0, :, :], x_input[:, 1, :, :]   # flatten_dataset(tf.make_ndarray(tf.make_tensor_proto(x_input))

    output_claim, encoded_state_claim, _ = self.rnn_1(c)
    del c
    context_vector_claim, attention_weights_claim = self.attention_claim(encoded_state_claim, output_claim)
    attention_claim = tf.concat([context_vector_claim, encoded_state_claim], axis=-1)

    output_evidence, encoded_state_evidence, _ = self.rnn_2(e)
    del e
    context_vector_evidence, attention_weights_evidence = self.attention_claim(encoded_state_evidence, output_evidence)
    attention_evidence = tf.concat([context_vector_evidence, encoded_state_evidence], axis=-1)
    
    x = self.merging_layer(attention_claim, attention_evidence)
    
    x = self.dense_1(x)

    x = self.dense_2(x)

    output = self.out(x)

    
    return output


rnn_first_attention = RNNFirstAttention(embedding_dimension, rnn_attention_unit_dense_1, rnn_attention_unit_dense_2, merging_type='sum')
rnn_first_attention.compile(optimizer='nadam',  # GRU adam:valaccuracy 71.4 vallos 98.9, sgd:valaccuracy 65.6 vallos 66.0, Nadam:valaccuracy 72.0 vallos 104.9, 
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_first_attention.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn_attention_batch_size,
          epochs=rnn_attention_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn_attention])

loss, accuracy = rnn_first_attention.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # 72.68, l2 72.7

"""### RNNSecond with Attention"""

class RNNSecond(tf.keras.Model):
  def __init__(self, enc_units, unit_dense_1, unit_dense_2, merging_type='concat', cosine_sim=False):
    super(RNNSecond, self).__init__()
    self.enc_units = enc_units
    self.rnn_1 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    self.rnn_2 = LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
    
    self.merging_layer = MergingLayer(merging_type, cosine_sim)

    self.attention_claim = AttentionRNN(self.enc_units)
    self.attention_evidence = AttentionRNN(self.enc_units)
    
    self.dense_1 = Dense(unit_dense_1, kernel_regularizer='l2')
    self.dense_2 = Dense(unit_dense_2, kernel_regularizer='l2')
    self.out = Dense(1, activation = 'sigmoid')

  def call(self, x_input):

    c, e = x_input[:, 0, :, :], x_input[:, 1, :, :] # flatten_dataset(tf.make_ndarray(tf.make_tensor_proto(x_input))

    output_vector_claim, encoded_state_claim, _ = self.rnn_1(c)
    del c
    
    output_vector_evidence, encoded_state_evidence, _ = self.rnn_2(e)  # [batch_size, max_length, embedding_dim]
    del e

    claim = tf.math.reduce_mean(output_vector_claim, axis=1)
    evidence = tf.math.reduce_mean(output_vector_evidence, axis=1)  # [batch_size, embedding_dim]

    context_vector_claim, attention_weights_claim = self.attention_claim(encoded_state_claim, claim)
    attention_claim = tf.concat([context_vector_claim, claim], axis=-1)

    context_vector_evidence, attention_weights_evidence = self.attention_claim(encoded_state_evidence, evidence)
    attention_evidence = tf.concat([context_vector_evidence, evidence], axis=-1)
    
    x = self.merging_layer(attention_claim, attention_evidence)
    
    x = self.dense_1(x)
    x = Dropout(0.3)(x)
    x = self.dense_2(x)
    x = Dropout(0.3)(x)
    output = self.out(x)

    return output


rnn_second_attention = RNNSecond(embedding_dimension, rnn_attention_unit_dense_1, rnn_attention_unit_dense_2, merging_type='concat')
rnn_second_attention.compile(optimizer='nadam',
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

rnn_second_attention.fit(X_train_encoded, Y_train_encoded, 
          batch_size=rnn_attention_batch_size,
          epochs=rnn_attention_epochs,
          validation_data=(X_validate_encoded, Y_validate_encoded),
          validation_steps=30,
          callbacks=[callback_early_stopping, callback_adaptive_lr, tensorboard_callback_rnn_attention])

loss, accuracy = rnn_second_attention.evaluate(X_validate_encoded, Y_validate_encoded)

print("Loss: ", loss)
print("Accuracy: ", accuracy)  # 71.5

"""# Tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/fit