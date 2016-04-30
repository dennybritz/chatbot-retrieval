import array
import numpy as np


def load_glove_vectors(filename, vocab=None):
    """
    Load glove vectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    dct = {}
    vectors = array.array('d')
    current_idx = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1
        word_dim = len(entries)
        num_vectors = len(dct)
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]


def evaluate_recall(y, y_labels, n=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_labels):
        if label in predictions[:n]:
            num_correct += 1
    return num_correct/num_examples
