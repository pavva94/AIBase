# -*- coding: utf-8 -*-
"""
We will see:

*   Building a vocabulary
*   Building a word-word co-occurrence matrix
*   Defining a similarity metric: cosine similarity
*   Embedding visualization and analysis of their semantic properties
*   Better sparse representations via PPMI weighting
*   Loading pre-trained dense word embeddings (Word2Vec, GloVe)
*   Checking out-of-vocabulary (OOV) terms
*   Handling OOV terms

"""
# system packages
import os
import shutil
import sys

# data and numerical management packages
import pandas as pd
import numpy as np

# useful during debugging (progress bars)
from tqdm import tqdm

# custom import
import tensorflow as tf

"""
[Part I] Sparse embeddings

"""

from urllib import request
import tarfile

# Config
print("Current work directory: {}".format(os.getcwd()))

dataset_folder = os.path.join(os.getcwd(), "Datasets")

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset_path = os.path.join(dataset_folder, "Movies.tar.gz")

print(dataset_path)

def download_dataset(download_path, url):
    if not os.path.exists(download_path):
        print("Downloading dataset...")
        request.urlretrieve(url, download_path)
        print("Download complete!")

def extract_dataset(download_path, extract_path):
    print("Extracting dataset... (it may take a while...)")
    with tarfile.open(download_path) as loaded_tar:
        loaded_tar.extractall(extract_path)
    print("Extraction completed!")

# Download
download_dataset(dataset_path, url)

# Extraction
extract_dataset(dataset_path, dataset_folder)

# Config
dataset_name = "aclImdb"
debug = True

def encode_dataset(dataset_name, debug=True):
    dataframe_rows = []

    for split in tqdm(['train', 'test']):
        for sentiment in ['pos', 'neg']:
            folder = os.path.join(os.getcwd(), "Datasets", dataset_name, split, sentiment)
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        # open the file
                        with open(file_path, mode='r', encoding='utf-8') as text_file:
                            # read it and extract informations
                            text = text_file.read()
                            score = filename.split("_")[1].split(".")[0]
                            file_id = filename.split("_")[0]

                            num_sentiment = -1

                            if sentiment == "pos" : num_sentiment = 1
                            elif sentiment == "neg" : num_sentiment = 0

                            # create single dataframe row
                            dataframe_row = {
                                "file_id": file_id,
                                "score": score,
                                "sentiment": num_sentiment,
                                "split": split,
                                "text": text
                            }

                            # print detailed info for the first file
                            if debug:
                                print(file_path)
                                print(filename)
                                print(file_id)
                                print(text)
                                print(score)
                                print(sentiment)
                                print(split)
                                print(dataframe_row)
                                debug = False
                            dataframe_rows.append(dataframe_row)

                except Exception as e:
                    print('Failed to process %s. Reason: %s' % (file_path, e))
                    sys.exit(0)

    folder = os.path.join(os.getcwd(), "Datasets", "Dataframes", dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # transform the list of rows in a proper dataframe
    df = pd.DataFrame(dataframe_rows)
    df = df[["file_id",
                        "score",
                        "sentiment",
                        "split",
                        "text"]]
    dataframe_path = os.path.join(folder, dataset_name + ".pkl")
    df.to_pickle(dataframe_path)

    return df


# Encoding
print("Encoding dataset...")
df = encode_dataset(dataset_name, debug)
print("Encoding completed!")

"""
Loading and Visualization

"""

# Inspection

print("Dataset size: {}".format(df.shape)) # (50000, 5)
print("Dataset columns: {}".format(df.columns.values)) # ['file_id', 'score', 'sentiment', 'split', 'text]

print("Classes distribution:\n{}".format(df.sentiment.value_counts())) # [0: 25000, 1: 25000]

print("Some examples: {}".format(df.iloc[:5]))


print(df.head)
print('First review text: \n{}'.format(df.iloc[0]['text']))
print('First review number of words: {}'.format(len(df.iloc[0]['text'].split(' '))))
print('First review number of unique words: {}'.format(len(set(df.iloc[0]['text'].split(' ')))))


"""
Building the Vocabulary

"""

import re
from functools import reduce
import nltk
from nltk.corpus import stopwords

# Config

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def lower(text):
    """
    Transforms given text to lower case.
    Example:
    Input: 'I really like New York city'
    Output: 'i really like new your city'
    """

    return text.lower()

def replace_special_characters(text):
    """
    Replaces special characters, such as paranthesis,
    with spacing character
    """

    return REPLACE_BY_SPACE_RE.sub(' ', text)

def filter_out_uncommon_symbols(text):
    """
    Removes any special character that is not in the
    good symbols list (check regular expression)
    """

    return GOOD_SYMBOLS_RE.sub('', text)

def remove_stopwords(text):
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])


def strip_text(text):
    """
    Removes any left or right spacing (including carriage return) from text.
    Example:
    Input: '  This assignment is cool\n'
    Output: 'This assignment is cool'
    """

    return text.strip()

PREPROCESSING_PIPELINE = [
                          lower,
                          replace_special_characters,
                          filter_out_uncommon_symbols,
                          remove_stopwords,
                          strip_text
                          ]

# Anchor method

def text_prepare(text, filter_methods=None):
    """
    Applies a list of pre-processing functions in sequence (reduce).
    Note that the order is important here!
    """

    filter_methods = filter_methods if filter_methods is not None else PREPROCESSING_PIPELINE

    return reduce(lambda txt, f: f(txt), filter_methods, text)

# Pre-processing

print('Pre-processing text...')

print()
print('[Debug] Before:\n{}'.format(df.text[:3]))
print()

# Replace each sentence with its pre-processed version
df['text'] = df['text'].apply(lambda txt: text_prepare(txt))

print('[Debug] After:\n{}'.format(df.text[:3]))
print()

print("Pre-processing completed!")


from collections import OrderedDict

# Function definition
def build_vocabulary(df):
    """
    Given a dataset, builds the corresponding word vocabulary.

    :param df: dataset from which we want to build the word vocabulary (pandas.DataFrame)
    :return:
      - word vocabulary: vocabulary index to word
      - inverse word vocabulary: word to vocabulary index
      - word listing: set of unique terms that build up the vocabulary
    """

    revs = []
    for rev in df['text']:
      revs = revs + tf.keras.preprocessing.text.text_to_word_sequence(rev, filters=[], lower=False)
    print(revs)
    unique_words = list(set(revs))
    #unique_words.sort()  # to make sure you get the same encoding at each run

    # Store them in a dict, associated with a numerical index
    word2idx = { word[1]: word[0] for word in enumerate(unique_words) }


    idx2word = { v: k for k, v in word2idx.items() }
    return idx2word, word2idx, unique_words


# Testing
idx_to_word, word_to_idx, word_listing = build_vocabulary(df)

print('[Debug] Index -> Word vocabulary size: {}'.format(len(idx_to_word)))
print('[Debug] Word -> Index vocabulary size: {}'.format(len(word_to_idx)))

print('[Debug] Some words: {}'.format([(idx_to_word[idx], idx) for idx in np.arange(10) + 1]))

# Evaluation

def evaluate_vocabulary(idx_to_word, word_to_idx, word_listing, df, check_default_size=False):

    # Check size
    print("[Vocabulary Evaluation] Size checking...")

    assert len(idx_to_word) == len(word_to_idx)
    assert len(idx_to_word) == len(word_listing)

    # Check content
    print("[Vocabulary Evaluation] Content checking...")

    for i in tqdm(range(0, len(idx_to_word))):
        assert idx_to_word[i] in word_to_idx
        assert word_to_idx[idx_to_word[i]] == i

    # Check consistency
    print("[Vocabulary Evaluation] Consistency checking...")

    _, _, first_word_listing = build_vocabulary(df)
    _, _, second_word_listing = build_vocabulary(df)
    assert first_word_listing == second_word_listing

    # Check toy example
    print("[Vocabulary Evaluation] Toy example checking...")
    toy_df = pd.DataFrame.from_dict({
        'text': ["all that glitters is not gold", "all in all i like this assignment"]
    })
    _, _, toy_word_listing = build_vocabulary(toy_df)
    toy_valid_vocabulary = set(' '.join(toy_df.text.values).split())
    assert set(toy_word_listing) == toy_valid_vocabulary


print("Vocabulary evaluation...")
evaluate_vocabulary(idx_to_word, word_to_idx, word_listing, df)
print("Evaluation completed!")


choice = input('Insert the word you want to check in the dictionary, or 0 to skip: ')
while choice != '0':
  print('Checking if the word selected is inside vocabulary: ')
  if choice in word_to_idx.keys():
    print('The word {} is present in the vocabulary with code: {}'.format(choice, word_to_idx[choice]))
  else:
    print('The word {} is NOT present in the vocabulary'.format(choice))

  choice = input('Insert the word you want to check in the dictionary, or 0 to skip: ')


import scipy.sparse    # defines several types of efficient sparse matrices
import zipfile
import gc
import requests
import time
import itertools

# Function definition

def co_occurrence_count(df, idx_to_word, word_to_idx, window_size=4):
    """
    Builds word-word co-occurrence matrix based on word counts.

    :param df: pre-processed dataset (pandas.DataFrame)
    :param idx_to_word: vocabulary map (index -> word) (dict)
    :param word_to_idx: vocabulary map (word -> index) (dict)

    :return
      - co_occurrence symmetric matrix of size |V| x |V| (|V| = vocabulary size)
    """
    ### YOUR CODE HERE ###
    matrix = scipy.sparse.lil_matrix((len(idx_to_word), len(idx_to_word)))

    # for every pair of words in the vocabulary i will count +1 when word_2 is 
    # found inside the window size centered in word_1
    # loop over the reviews
    for text in df['text']:
        words = text.split()
        words = [word_to_idx[x] for x in words]

        for i in range(0, len(words)):  # 1, 2, 3-..
          current = words[i]
          window = []

          for j in range(2*window_size+1):  # J=[0, 8]
            j = j-window_size  # j=[-4,4]

            if j!=0 and i+j >= 0 and i+j <= len(words)-1 and current != words[i+j]:
              window.append(words[i+j])

          for id_word in window:
              matrix[current, id_word] += 1

    return matrix


# Testing
window_size = 4

# Clean RAM before re-running this code snippet to avoid session crash
if 'co_occurrence_matrix' in globals():
    del co_occurrence_matrix
    gc.collect()
    time.sleep(10.)

print("Building co-occurrence count matrix... (it may take a while...)")
co_occurrence_matrix = co_occurrence_count(df, idx_to_word, word_to_idx, window_size)
print("Building completed!")

# Evaluation

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_toy_data(benchmark_path):
    toy_data_path = os.path.join(benchmark_path, 'co-occurrence_count_benchmark.zip')
    toy_data_url_id = "1z8qp034utvW7kv-9Q_TACJv3_sdCzkZg"
    toy_url = "https://docs.google.com/uc?export=download"

    if not os.path.exists(benchmark_path):
        os.makedirs(benchmark_path)

    if not os.path.exists(toy_data_path):
        print("Downloading co-occurrence count matrix benchmark data...")
        with requests.Session() as current_session:
            response = current_session.get(toy_url,
                                   params={'id': toy_data_url_id},
                                   stream=True)
        save_response_content(response, toy_data_path)
        print("Download complete!")

        print("Extracting dataset...")
        with zipfile.ZipFile(toy_data_path) as loaded_zip:
            loaded_zip.extractall(benchmark_path)
        print("Extraction complete!")

def evaluate_co_occurrence_matrix(matrix):
    is_sparse = False

    if hasattr(scipy.sparse, type(matrix).__name__):
        print("Detected sparse co-occurrence matrix!")
        is_sparse = True

    # Check symmetry
    print("[Co-occurrence count matrix Evaluation] Symmetry checking...")
    if is_sparse:
        assert (matrix != matrix.transpose()).nnz == 0
    else:
        assert np.equal(matrix, matrix.transpose()).all()

    # Check toy example
    print("[Co-occurrence count matrix Evaluation] Toy example checking...")
    toy_df = pd.DataFrame.from_dict({
        'text': ["all that glitters is not gold",
                 "all in all i like this assignment"],
    })
    benchmark_path = os.path.join(os.getcwd(), 'Benchmark')
    toy_path = os.path.join(benchmark_path, 'co-occurrence_count_benchmark')
    download_toy_data(benchmark_path)

    toy_idx_to_word = np.load(os.path.join(toy_path, 'toy_idx_to_word.npy'), allow_pickle=True).item()
    toy_word_to_idx = np.load(os.path.join(toy_path, 'toy_word_to_idx.npy'), allow_pickle=True).item()

    toy_matrix = co_occurrence_count(toy_df, toy_idx_to_word, toy_word_to_idx, window_size=1)
    toy_valid_matrix = np.load(os.path.join(toy_path, 'toy_co_occurrence_matrix_count.npy'))

    if is_sparse:
        assert np.equal(toy_matrix.todense(), toy_valid_matrix).all()
    else:
        assert np.equal(toy_matrix, toy_valid_matrix).all()


print("Evaluating co-occurrence matrix")
evaluate_co_occurrence_matrix(co_occurrence_matrix)
print("Evaluation completed!")


"""
Embedding Visualization

"""

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Function definition

def visualize_embeddings(embeddings, word_annotations=None, word_to_idx=None):
    """
    Plots given reduce word embeddings (2D).
    Users can highlight specific words (word_annotations list) in order to better
    analyse the effectiveness of the embedding method.

    :param embeddings: word embedding matrix of shape (words, 2) retrieved via a
                       dimensionality reduction technique.
    :param word_annotations: list of words to be annotated.
    :param word_to_idx: vocabulary map (word -> index) (dict)
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    if word_annotations:
        print("Annotating words: {}".format(word_annotations))

        word_indexes = []
        for word in word_annotations:
            word_index = word_to_idx[word]
            word_indexes.append(word_index)

        word_indexes = np.array(word_indexes)

        other_embeddings = embeddings[np.setdiff1d(np.arange(embeddings.shape[0]), word_indexes)]
        target_embeddings = embeddings[word_indexes]

        ax.scatter(other_embeddings[:, 0], other_embeddings[:, 1], alpha=0.1, c='blue')
        ax.scatter(target_embeddings[:, 0], target_embeddings[:, 1], alpha=1.0, c='red')
        ax.scatter(target_embeddings[:, 0], target_embeddings[:, 1], alpha=1, facecolors='none', edgecolors='r', s=1000)

        for word, word_index in zip(word_annotations, word_indexes):
            word_x, word_y = embeddings[word_index, 0], embeddings[word_index, 1]
            ax.annotate(word, xy=(word_x, word_y))

    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.1, c='blue')

    # Set proper axis limit range
    # We avoid outliers ruining the visualization if they are quite far away
    xmin_quantile = np.quantile(embeddings[:, 0], q=0.01)
    xmax_quantile = np.quantile(embeddings[:, 0], q=0.99)

    ymin_quantile = np.quantile(embeddings[:, 1], q=0.01)
    ymax_quantile = np.quantile(embeddings[:, 1], q=0.99)

    ax.set_xlim(xmin_quantile, xmax_quantile)
    ax.set_ylim(ymin_quantile, ymax_quantile)


def reduce_SVD(embeddings):
    """
    Applies SVD dimensionality reduction.

    :param embeddings: word embedding matrix of shape (words, dim). In the case
                       of a word-word co-occurrence matrix the matrix shape would
                       be (words, words).

    :return
        - 2-dimensional word embedding matrix of shape (words, 2)
    """
  
    print("Running SVD reduction method...")
    svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
    reduced = svd.fit_transform(embeddings)
    print("SVD reduction completed!")

    return reduced

# Note: this method may take a while
def reduce_tSNE(embeddings):
    """
    Applies t-SNE dimensionality reduction.

    :param embeddings: word embedding matrix of shape (words, dim). In the case
                       of a word-word co-occurrence matrix the matrix shape would
                       be (words, words).

    :return
        - 2-dimensional word embedding matrix of shape (words, 2)
    """

    print("Running t-SNE reduction method... (it may take a while...)")
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000, metric='cosine', n_jobs=2)
    reduced = tsne.fit_transform(embeddings)
    print("t-SNE reduction completed!")
    print(reduced.shape)

    return reduced

# Testing

# SVD
reduced_SVD = reduce_SVD(co_occurrence_matrix)
visualize_embeddings(reduced_SVD, ['good', 'love', 'beautiful'], word_to_idx)

# t-SNE
# Note: this method may take a while (just relax :-))
reduced_tSNE = reduce_tSNE(co_occurrence_matrix)
visualize_embeddings(reduced_tSNE, ['good', 'love', 'beautiful'], word_to_idx)

plt.show()

# SVD
reduced_SVD = reduce_SVD(co_occurrence_matrix)
visualize_embeddings(reduced_SVD, ['music', 'melody', 'sound'], word_to_idx)

# t-SNE
# Note: this method may take a while (just relax :-))
reduced_tSNE = reduce_tSNE(co_occurrence_matrix)
visualize_embeddings(reduced_tSNE, ['music', 'melody', 'sound'], word_to_idx)

plt.show()

"""
Embedding properties

"""

# Function definition

from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity as cs

def cosine_similarity(p, q, transpose_p=False, transpose_q=False):
    """
    Computes the cosine similarity of two d-dimensional matrices

    :param p: d-dimensional vector (np.ndarray) of shape (p_samples, d)
    :param q: d-dimensional vector (np.ndarray) of shape (q_samples, d)
    :param transpose_p: whether to transpose p or not
    :param transpose_q: whether to transpose q or not

    :return
        - cosine similarity matrix S of shape (p_samples, q_samples)
          where S[i, j] = s(p[i], q[j])
    """

    # If it is a vector, consider it as a single sample matrix
    if len(p.shape) == 1:
        p = p.reshape(1, -1)
    if len(q.shape) == 1:
        q = q.reshape(1, -1)

    # cosine similarity: sum(pi,qi)/(sqrt(sum(a^2))*sqrt(sum(a^2)))
    '''if transpose_p:
      p = np.transpose(p)
    if transpose_q:
      q = np.transpose(q)    
    '''

    '''
    matrix = scipy.sparse.lil_matrix((p.shape[0], q.shape[0]))

    for i, pi in enumerate(p):
      for j, qj in enumerate(q):
        n = sum([a*b for a,b in zip(pi,qj)])
        d1 = sqrt(sum(np.array(list(map(lambda x: x*x, pi)))))
        d2 = sqrt(sum(np.array(list(map(lambda x: x*x, qj)))))

        matrix[i,j] = n/(d1*d2)
    '''
    matrix = cs(p, q)

    return matrix

# Testing

print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(co_occurrence_matrix,
                                      co_occurrence_matrix,
                                      transpose_q=True)
print("Similarity completed!")

# Evaluation

def sparse_allclose(a, b, rtol=1e-5, atol = 1e-8):
    c = np.abs(np.abs(a - b) - rtol * np.abs(b))
    return c.max() <= atol

def evaluate_cosine_similarity(similarity_matrix):

    # Vector similarity
    print('[Cosine similarity Evaluation] Vector similarity check...')

    p = np.array([5., 6., 0.3, 1.])
    q = np.array([50., 6., 0., 0.])
    assert np.allclose([[0.72074324]], cosine_similarity(p, q, transpose_q=True))

    # Matrix similarity
    print('[Cosine similarity Evaluation] Matrix similarity check...')

    toy_matrix = np.array([5., 6., 0.3, 1.,
                           50., 6., 0., 0.,
                           0., 100., 20., 4.]).reshape(3, 4)
    true_matrix = np.array([1., 0.72074324, 0.75852259,
                            0.72074324, 1., 0.11674173,
                            0.75852259, 0.11674173, 1.]).reshape(3, 3)
    proposed_matrix = cosine_similarity(toy_matrix, toy_matrix, transpose_q=True)
    
    assert np.allclose(proposed_matrix, true_matrix)

    # There might be some numerical error that invalidates the np.equal check
    assert np.allclose(proposed_matrix, proposed_matrix.transpose())

    # Check symmetry
    print("[Cosine similarity Evaluation] Symmetry checking...")

    is_sparse = False

    if hasattr(scipy.sparse, type(similarity_matrix).__name__):
        print("Detected sparse cosine similarity matrix!")
        is_sparse = True

    if is_sparse:
        try:
            assert (similarity_matrix != similarity_matrix.transpose()).nnz == 0
        except AssertionError:
            assert sparse_allclose(similarity_matrix, similarity_matrix.transpose())
    else:
        # There might be some numerical error that invalidates the np.equal check
        assert np.allclose(similarity_matrix, similarity_matrix.transpose())

print('Evaluating cosine similarity...')
evaluate_cosine_similarity(similarity_matrix)
print('Evaluation completed!')

print("Example 1: Love")
love = word_to_idx["love"]
dislike = word_to_idx["dislike"]
like = word_to_idx["like"]

#Let's try to analyze synonyms
love_vector = co_occurrence_matrix[love]
like_vector = co_occurrence_matrix[like]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(love_vector, like_vector)))

#Let's try to analyze synonyms
dislike_vector = co_occurrence_matrix[dislike]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(love_vector, dislike_vector)))


print("Example 2: Good")
good = word_to_idx["good"]
bad = word_to_idx["bad"]
fine = word_to_idx["fine"]

#Let's try to analyze synonyms
good_vector = co_occurrence_matrix[good]
fine_vector = co_occurrence_matrix[fine]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(good_vector, fine_vector)))

#Let's try to analyze synonyms
bad_vector = co_occurrence_matrix[bad]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(good_vector, bad_vector)))
print("Co-occurence of good and fine {}".format(co_occurrence_matrix[good, fine]))
print("Co-occurence of good and bad {}".format(co_occurrence_matrix[good, bad]))


def get_top_K_indexes(data, K):
    """
    Returns the top K indexes of a 1-dimensional array (descending order)
    Example:
        data = [0, 7, 2, 1]
        best_indexes:
        K = 1 -> [1] (data[1] = 7)
        K = 2 -> [1, 2]
        K = 3 -> [1, 2, 3]
        K = 4 -> [1, 2, 3, 4]

    :param data: 1-d dimensional array
    :param K: number of highest value elements to consider

    :return
        - array of indexes corresponding to elements of highest value
    """
    best_indexes = np.argsort(data, axis=0)[::-1]
    best_indexes = best_indexes[:K]

    return best_indexes

def get_top_K_word_ranking(embedding_matrix, idx_to_word, word_to_idx,
                           positive_listing, negative_listing, K):
    """
    Finds the top K most similar words following this reasoning:
        1. words that have highest similarity to words in positive_listing
        2. words that have highest distance to words in negative_listing
    
    Positive and negative listing can be defined accordingly to a given analogy
    Example:
        
        man : king :: woman : x
    
    positive_listing = ['king', 'woman']
    negative_listing = ['man']

    This is equivalent to: compute king - man + woman, and then find the
    most similar candidate.
    
    :param embedding_matrix: embedding matrix of shape (words, embedding dimension).
    Note that in the case of a co-occurrence matrix, the shape is (words, words).
    :param idx_to_word: vocabulary map (index -> word) (dict)
    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param positive_listing: list of words that should have high similarity with
                             top K retrieved ones.
    :param negative_listing: list of words that should have high distance to
                             top K retrieved ones.
    :param K: number of best word matches to consider

    :return
        - top K word matches according to aforementioned criterium
        - similarity values of top K word matches according to aforementioned
          criterium
    """


    # Positive words (similarity)
    positive_indexes = np.array([word_to_idx[word] for word in positive_listing])
    word_positive_vector = np.sum(embedding_matrix[positive_indexes, :], axis=0)

    # Negative words (distance)
    negative_indexes = np.array([word_to_idx[word] for word in negative_listing])
    word_negative_vector = np.sum(embedding_matrix[negative_indexes, :], axis=0)

    # Find candidate words
    target_vector = (word_positive_vector - word_negative_vector) / (len(positive_listing) + len(negative_listing))
    total_indexes = np.concatenate((positive_indexes, negative_indexes))
    valid_indexes = np.setdiff1d(np.arange(similarity_matrix.shape[0]), total_indexes)
    candidate_vectors = embedding_matrix[valid_indexes]

    candidate_similarities = cosine_similarity(candidate_vectors, target_vector, transpose_q=True)
    candidate_similarities = candidate_similarities.ravel()

    relative_indexes = get_top_K_indexes(candidate_similarities, K)
    top_K_indexes = valid_indexes[relative_indexes]
    top_K_words = [idx_to_word[idx] for idx in top_K_indexes]

    return top_K_words, candidate_similarities[relative_indexes]

"""Now do it yourself! Find some examples of analogies that hold and other that do not. Remember to give a proper explanation concerning obtained results.

**Note**: 1-2 examples are sufficient. This exercies is just another way to inspect word embeddings.
"""

K = 5
# Example analogy: tv : episodes :: film : x
# positive listing -> [episodes, film]
# negative listing ->  [tv]
# masterpiece : superb :: x : tragic

'''
top_K_words, top_K_values = get_top_K_word_ranking(co_occurrence_matrix,
                                                   idx_to_word,
                                                   word_to_idx,
                                                   ['good', 'scenes'],
                                                   ['tragic'],
                                                   K)
'''
positive_listing = ['king', 'man']
negative_listing = ['woman']
top_K_words, top_K_values = get_top_K_word_ranking(co_occurrence_matrix,
                                                   idx_to_word,
                                                   word_to_idx,
                                                   positive_listing,
                                                   negative_listing,
                                                   K)
print('Top K words: ', top_K_words)
print('Top K values: ', top_K_values)

positive_listing = ['doctor', 'woman']
negative_listing = ['man']
top_K_words, top_K_values = get_top_K_word_ranking(co_occurrence_matrix,
                                                   idx_to_word,
                                                   word_to_idx,
                                                   positive_listing,
                                                   negative_listing,
                                                   K)
print('Top K words: ', top_K_words)
print('Top K values: ', top_K_values)


positive_listing = ['emergency', 'happiness']
negative_listing = ['fear']
top_K_words, top_K_values = get_top_K_word_ranking(co_occurrence_matrix,
                                                   idx_to_word,
                                                   word_to_idx,
                                                   positive_listing,
                                                   negative_listing,
                                                   K)

print('Top K words: ', top_K_words)
print('Top K values: ', top_K_values)

positive_listing = ['housewife', 'man']
negative_listing = ['woman']
top_K_words, top_K_values = get_top_K_word_ranking(co_occurrence_matrix,
                                                   idx_to_word,
                                                   word_to_idx,
                                                   positive_listing,
                                                   negative_listing,
                                                   K)
print('Top K words: ', top_K_words)
print('Top K values: ', top_K_values)


# Function definition
import math

def convert_ppmi(co_occurrence_matrix):
    """
    Converts a count-based co-occurrence matrix to a PPMI matrix

    :param co_occurrence_matrix: count based co-occurrence matrix of shape (|V|, |V|)
    
    :return
        - PPMI co-occurrence matrix of shape (|V|, |V|)
    """

    ppmi_matrix = scipy.sparse.lil_matrix((co_occurrence_matrix.shape[0], co_occurrence_matrix.shape[1]))
    ppmi_denominator = co_occurrence_matrix.sum()
    print("ppmi_denominator {}".format(ppmi_denominator))
    for i in range(0, ppmi_matrix.shape[0]):
      pi_star = co_occurrence_matrix[i,:].sum()/ppmi_denominator
      #print("pi_star {}".format(pi_star))
      for j in range(0, ppmi_matrix.shape[1]):
        pj_star = co_occurrence_matrix[:,j].sum()/ppmi_denominator
        #print("pj_star {}".format(pj_star))
        pij = co_occurrence_matrix[i,j]/ppmi_denominator
        #print("pij {}".format(pij))
        #print("pij/pi_star*pj_star {}".format(pij/(pi_star*pj_star)))
        if pij !=0:
          ppmi_matrix[i,j] = max(math.log2(pij/(pi_star*pj_star)), 0)
        else:
          ppmi_matrix[i,j] = 0

    return ppmi_matrix


# Testing

print("Computing PPMI co-occurrence matrix...")
ppmi_occurrence_matrix = convert_ppmi(co_occurrence_matrix)
print("PPMI completed!")

# Evaluation

def evaluate_ppmi_matrix(matrix):
    is_sparse = False

    if hasattr(scipy.sparse, type(matrix).__name__):
        print("Detected sparse PPMI co-occurrence matrix!")
        is_sparse = True

    # Check symmetry
    print("[Co-occurrence PPMI matrix Evaluation] Symmetry checking...")
    if is_sparse:
        try:
            assert (matrix != matrix.transpose()).nnz == 0
        except AssertionError:
            assert sparse_allclose(matrix, matrix.transpose())
    else:
        try:
            assert np.equal(matrix, matrix.transpose()).all()
        except AssertionError:
            assert np.allclose(matrix, matrix.transpose())

    # A very simple example
    print("[Co-occurrence PPMI matrix Evaluation] Toy example checking...")

    toy_df = pd.DataFrame.from_dict({
        'sentence_1': ["All that glitters is not gold"],
        'sentence_2': ["All in all I like this assignment"]
    })

    # We should already have download co-occurrence benchmark data
    benchmark_path = os.path.join(os.getcwd(), 'Benchmark')
    toy_path = os.path.join(benchmark_path, 'co-occurrence_count_benchmark')
    toy_idx_to_word = np.load(os.path.join(toy_path, 'toy_idx_to_word.npy'), allow_pickle=True).item()
    toy_word_to_idx = np.load(os.path.join(toy_path, 'toy_word_to_idx.npy'), allow_pickle=True).item()
    toy_valid_matrix = np.load(os.path.join(toy_path, 'toy_co_occurrence_matrix_count.npy'))

    toy_ppmi_matrix = convert_ppmi(toy_valid_matrix)
    toy_valid_ppmi_matrix = np.load(os.path.join(toy_path, 'toy_co_occurrence_matrix_ppmi.npy'))

    if is_sparse:
        try:
            assert (toy_ppmi_matrix != toy_valid_ppmi_matrix).nnz == 0
        except AssertionError:
            assert sparse_allclose(toy_ppmi_matrix, toy_valid_ppmi_matrix)
    else:
        try:
            assert np.equal(toy_ppmi_matrix, toy_valid_ppmi_matrix).all()
        except AssertionError:
            assert np.allclose(toy_ppmi_matrix, toy_valid_ppmi_matrix)


print('Evaluating PPMi matrix conversion...')
evaluate_ppmi_matrix(ppmi_occurrence_matrix)
print('Evaluation completed!')


"""
Visualization (cont'd)

"""

# SVD
reduced_SVD = reduce_SVD(ppmi_occurrence_matrix)
visualize_embeddings(reduced_SVD, ['good', 'love', 'beautiful'], word_to_idx)

# t-SNE
# Note: this method may take a while (just relax :-))
reduced_tSNE = reduce_tSNE(ppmi_occurrence_matrix)
visualize_embeddings(reduced_tSNE, ['good', 'love', 'beautiful'], word_to_idx)

plt.show()


visualize_embeddings(reduced_SVD, ['amazing', 'scene', 'movie'], word_to_idx)
visualize_embeddings(reduced_tSNE, ['amazing', 'scene', 'movie'], word_to_idx)

plt.show()

print("Example 1: Fast")
good = word_to_idx["good"]
evil = word_to_idx["evil"]
positive = word_to_idx["positive"]

#Let's try to analyze synonyms
good_vector = ppmi_occurrence_matrix[good]
positive_vector = ppmi_occurrence_matrix[positive]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(good_vector, positive_vector)))

#Let's try to analyze antonyms
evil_vector = ppmi_occurrence_matrix[evil]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(good_vector, evil_vector)))

### YOUR CODE HERE  ###
print("Example 2: Good")
original = word_to_idx["original"]
conventional = word_to_idx["conventional"]
fresh = word_to_idx["fresh"]

#Let's try to analyze synonyms
original_vector = ppmi_occurrence_matrix[original]
fresh_vector = ppmi_occurrence_matrix[fresh]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(original_vector, fresh_vector)))

#Let's try to analyze synonyms
conventional_vector = ppmi_occurrence_matrix[conventional]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(original_vector, conventional_vector)))

"""
[Part II] Dense embeddings

"""

import gensim
import gensim.downloader as gloader

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
# Word2Vec -> 300
embedding_model_type = "glove"
embedding_dimension = 50

embedding_model = load_embedding_model(embedding_model_type, embedding_dimension)

"""
Out of vocabulary (OOV) words

"""

# Function definition

print(embedding_model)

def check_OOV_terms(embedding_model, word_listing):
    """
    Checks differences between pre-trained embedding model vocabulary
    and dataset specific vocabulary in order to highlight out-of-vocabulary terms.

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_listing: dataset specific vocabulary (list)

    :return
        - list of OOV terms
    """

    ### YOUR CODE HERE ###
    model_vocabulary = embedding_model.vocab
    oov_terms = list(set(model_vocabulary) - set(word_listing))
    return oov_terms


oov_terms = check_OOV_terms(embedding_model, word_listing)

print("Total OOV terms: {0} ({1:.2f}%)".format(len(oov_terms), float(len(oov_terms)) / len(word_listing)))
print("First 5 words OOV: {} ".format((oov_terms[:5])))
print('plutocracy' in word_listing)


# Function definition

def build_embedding_matrix(embedding_model, embedding_dimension, word_to_idx, oov_terms):
    """
    Builds the embedding matrix of a specific dataset given a pre-trained word embedding model

    :param embedding_model: pre-trained word embedding model (gensim wrapper)
    :param word_to_idx: vocabulary map (word -> index) (dict)
    :param oov_terms: list of OOV terms (list)
    :param co_occorruence_count_matrix: the co-occurrence count matrix of the given dataset (window size 1)

    :return
        - embedding matrix that assigns a high dimensional vector to each word in the dataset specific vocabulary (shape |V| x d)
    """

    ### YOUR CODE HERE ###
    V = len(word_to_idx)
    D = embedding_dimension
    embedding_matrix = np.zeros(shape=(V,D)) 
    word_listing = list(word_to_idx.keys())
    for i in range(len(word_to_idx)):
      curr_word = word_listing[i]
      if curr_word in embedding_model.vocab:
        embedding_matrix[i] = embedding_model[curr_word]
      else:
        embedding_matrix[i] = np.zeros(shape=D)
    for j in range(V):
      if not (embedding_matrix[j].any()):
        if j != 0 and j < V:
          embedding_matrix[j] = np.mean([embedding_matrix[j-1],embedding_matrix[j+1]])

    return embedding_matrix

# Testing

embedding_matrix = build_embedding_matrix(embedding_model, embedding_dimension, word_to_idx, oov_terms)

print("Embedding matrix shape: {}".format(embedding_matrix.shape))

"""
Embedding visualization (cont'd)

"""


# SVD
reduced_SVD = reduce_SVD(embedding_matrix)
visualize_embeddings(reduced_SVD, ['good', 'love', 'beautiful'], word_to_idx)

# t-SNE
# Note: this method may take a while (just relax :-))
reduced_tSNE = reduce_tSNE(embedding_matrix)
visualize_embeddings(reduced_tSNE, ['good', 'love', 'beautiful'], word_to_idx)

plt.show()

# SVD
reduced_SVD = reduce_SVD(embedding_matrix)
visualize_embeddings(reduced_SVD, ['movie', 'cinema', 'actor'], word_to_idx)

# t-SNE
# Note: this method may take a while (just relax :-))
reduced_tSNE = reduce_tSNE(embedding_matrix)
visualize_embeddings(reduced_tSNE, ['movie', 'cinema', 'actors'], word_to_idx)

plt.show()

print("Example 1: Actor")
actor = word_to_idx["actor"]
tree = word_to_idx["tree"]
character = word_to_idx["character"]

#Let's try to analyze synonyms
actor_vector = embedding_matrix[actor]
character_vector = embedding_matrix[character]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(actor_vector, character_vector)))

#Let's try to analyze antonyms
tree_vector = embedding_matrix[tree]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(actor_vector, tree_vector)))

print("Example 2: Good")
good = word_to_idx["good"]
bad = word_to_idx["bad"]
great = word_to_idx["great"]

#Let's try to analyze synonyms
good_vector = embedding_matrix[good]
great_vector = embedding_matrix[great]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(good_vector, great_vector)))

#Let's try to analyze synonyms
bad_vector = embedding_matrix[bad]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(good_vector, bad_vector)))

print("Example 3: Love")
love = word_to_idx["love"]
dislike = word_to_idx["dislike"]
like = word_to_idx["like"]

#Let's try to analyze synonyms
love_vector = embedding_matrix[love]
like_vector = embedding_matrix[like]

print("Cosine Similarity between Synonyms: {}".format(cosine_similarity(love_vector, like_vector)))

#Let's try to analyze synonyms
dislike_vector = embedding_matrix[dislike]

print("Cosine Similarity between Antonyms: {}".format(cosine_similarity(love_vector, dislike_vector)))
