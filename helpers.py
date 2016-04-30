import array
import numpy as np
import pandas as pd


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


def convert_to_labeled_df(df):
    """
    Converts the test/validation data from the Ubuntu Dialog corpus into a train-like Data Frame with labels.
    This Data Frame can be used to easily get accuarcy values for cross-validation
    """
    result = []
    for idx, row in df.iterrows():
        context = row.Context
        result.append([context, row.iloc[1], 1])
        for distractor in row.iloc[2:]:
            result.append([context, distractor, 0])
    return pd.DataFrame(result, columns=["Context", "Utterance", "Label"])
