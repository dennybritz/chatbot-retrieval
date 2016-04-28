import numpy as np
import pickle
import pandas as pd
import os
import tensorflow as tf
from tensorflow.contrib import skflow
from tensorflow.python.framework import dtypes
import array



def load_glove_vectors(filename, vocab=None):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    dct = {}
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tokens = line.split(" ")
            word = str(tokens[0])
            entries = tokens[1:]
            dct[word] = list(np.array(entries, dtype=float))
            #dct[word] = entries
            if idx%10000 == 0:
                print(idx)
        return dct


# The maximum number of words to consider for the contexts
MAX_CONTEXT_LENGTH = 80
# The maximum number of words to consider for the utterances
MAX_UTTERANCE_LENGTH = 40
# Word embedding dimensionality
EMBEDDING_DIM = 300
# data dir
data_dir = "./data/"
# glove dir
glove_dir = "./glove/"


# Load Data
# ==================================================
print("Loading data...")
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
validation_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
y_test = np.zeros(len(test_df))
print("Loading glove embedding...")
#glove_w2v = load_vec(os.path.join(data_dir, "vectors.txt"))
glove_w2v = load_glove_vectors(os.path.join(glove_dir, "glove.840B.300d.txt"))

# Preprocessing
# ==================================================
# Create vocabulary mapping
all_sentences = np.append(train_df.Context, train_df.Utterance)
vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_CONTEXT_LENGTH)
vocab_processor.fit(all_sentences)

print("start preparing word embedding...")

WE = []
cover = 0
#mapping_list = {"<unk>":0}
dataset_vocab = vocab_processor.vocabulary_._reverse_mapping
for i in range(len(dataset_vocab)):
    word = str(dataset_vocab[i])
    if glove_w2v.get(word) == None:
        WE.append(np.random.uniform(-0.1,0.1,EMBEDDING_DIM))
        continue
    WE.append(glove_w2v[word])
    cover += 1
     # mapping_list[word] = i

WE = np.array(WE)
print(type(WE))
print(WE.shape)
print("total word: "+str((WE.shape[0])))
print("cover word: "+str(cover))

pickle.dump(WE, open(os.path.join(data_dir,"embedding.p"), "wb"))
#pickle.dump(mapping_list, open("mappingList.p", "wb"))

