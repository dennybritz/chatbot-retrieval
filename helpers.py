import array
import numpy as np


def load_glove_vectors(filename, vocab=None):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    dct = {}
    vectors = array.array('d')
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                dct[word] = idx
                vectors.extend(float(x) for x in entries)
        word_dim = len(entries)
        num_vectors = len(dct)
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]
