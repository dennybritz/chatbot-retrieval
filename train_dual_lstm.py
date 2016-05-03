#! /usr/bin/env python

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.contrib import skflow
from tensorflow.python.framework import dtypes
from helpers import load_glove_vectors, evaluate_recall

# Learning Parameters
tf.flags.DEFINE_integer("num_steps", 1000000, "Number of training steps")
tf.flags.DEFINE_integer("batch_size", 256, "Batch size")
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning Rate")
tf.flags.DEFINE_float("learning_rate_decay_rate", 0.95, "Learning Rate Decay Factor")
tf.flags.DEFINE_integer("learning_rate_decay_every", 2000, "Decay after this many steps")
tf.flags.DEFINE_string("optimizer", "Adagrad", "Optimizer (Adam, Adagrad or SGD)")

# Model Parameters
tf.flags.DEFINE_integer("max_content_length", 120, "Maximum length of context in words")
tf.flags.DEFINE_integer("max_utterance_length", 40, "Maximum length of utterance in word")
tf.flags.DEFINE_integer("embedding_dim", 300, "Embedding dimensionality")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of RNN/LSTM state")

# Data
tf.flags.DEFINE_string("data_dir", "./data", "Data directory that contain train/valid/test CSVs")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# The maximum number of words to consider for the contexts
MAX_CONTEXT_LENGTH = FLAGS.max_content_length
# The maximum number of words to consider for the utterances
MAX_UTTERANCE_LENGTH = FLAGS.max_utterance_length
# Word embedding dimensionality
EMBEDDING_DIM = FLAGS.embedding_dim
# LSTM Cell dimensionality
RNN_DIM = FLAGS.rnn_dim


# Load Data
# ==================================================
print("Loading data...")
train_df = pd.read_csv(os.path.join(FLAGS.data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(FLAGS.data_dir, "test.csv"))
validation_df = pd.read_csv(os.path.join(FLAGS.data_dir, "valid.csv"))
y_test = np.zeros(len(test_df))


# Preprocessing
# ==================================================
# Create vocabulary mapping
all_sentences = np.append(train_df.Context, train_df.Utterance)
vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_CONTEXT_LENGTH, min_frequency=5)
vocab_processor.fit(all_sentences)

# Transform contexts and utterances
X_train_context = np.array(list(vocab_processor.transform(train_df.Context)))
X_train_utterance = np.array(list(vocab_processor.transform(train_df.Utterance)))

# Generate training tensor
X_train = np.stack([X_train_context, X_train_utterance], axis=1)
y_train = train_df.Label

n_words = len(vocab_processor.vocabulary_)
print("Total words: {}".format(n_words))


# Load glove vectors
# ==================================================
vocab_set = set(vocab_processor.vocabulary_._mapping.keys())
glove_vectors, glove_dict = load_glove_vectors(os.path.join(FLAGS.data_dir, "glove.840B.300d.txt"), vocab_set)


# Build initial word embeddings
# ==================================================
initial_embeddings = np.random.uniform(-0.1, 0.1, (n_words, EMBEDDING_DIM)).astype("float32")
for word, glove_word_idx in glove_dict.items():
    word_idx = vocab_processor.vocabulary_.get(word)
    initial_embeddings[word_idx, :] = glove_vectors[glove_word_idx]


# Define RNN Dual Encoder Model
# ==================================================

def get_sequence_length(input_tensor, max_length):
    """
    If a sentence is padded, returns the index of the first 0 (the padding symbol).
    If the sentence has no padding, returns the max length.
    """
    zero_tensor = np.zeros_like(input_tensor)
    comparsion = tf.equal(input_tensor, zero_tensor)
    zero_positions = tf.argmax(tf.to_int32(comparsion), 1)
    position_mask = tf.to_int64(tf.equal(zero_positions, 0))
    sequence_lengths = zero_positions + position_mask * max_length
    return sequence_lengths


def rnn_encoder_model(X, y):
    # Split input tensor into separare context and utterance tensor
    context, utterance = tf.split(1, 2, X, name='split')
    context = tf.squeeze(context, [1])
    utterance = tf.squeeze(utterance, [1])
    utterance_truncated = tf.slice(utterance, [0, 0], [-1, MAX_UTTERANCE_LENGTH])

    # Calculate the sequence length for RNN calculation
    context_seq_length = get_sequence_length(context, MAX_CONTEXT_LENGTH)
    utterance_seq_length = get_sequence_length(utterance, MAX_UTTERANCE_LENGTH)

    # Embed context and utterance into the same space
    with tf.variable_scope("shared_embeddings") as vs, tf.device('/cpu:0'):
        embedding_tensor = tf.convert_to_tensor(initial_embeddings)
        embeddings = tf.get_variable("word_embeddings", initializer=embedding_tensor)
        # Embed the context
        word_vectors_context = skflow.ops.embedding_lookup(embeddings, context)
        word_list_context = skflow.ops.split_squeeze(1, MAX_CONTEXT_LENGTH, word_vectors_context)
        # Embed the utterance
        word_vectors_utterance = skflow.ops.embedding_lookup(embeddings, utterance_truncated)
        word_list_utterance = skflow.ops.split_squeeze(1, MAX_UTTERANCE_LENGTH, word_vectors_utterance)

    # Run context and utterance through the same RNN
    with tf.variable_scope("shared_rnn_params") as vs:
        cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_DIM)
        context_outputs, _ = tf.nn.rnn(
            cell, word_list_context, dtype=dtypes.float32, sequence_length=context_seq_length)
        encoding_context = context_outputs[-1]
        vs.reuse_variables()
        encoding_outputs, _ = tf.nn.rnn(
            cell, word_list_utterance, dtype=dtypes.float32, sequence_length=utterance_seq_length)
        encoding_utterance = encoding_outputs[-1]

    with tf.variable_scope("prediction") as vs:
        W = tf.get_variable("W",
                            shape=[encoding_context.get_shape()[1], encoding_utterance.get_shape()[1]],
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable("b", [1])

        # We can interpret this is a "Generated context"
        generated_context = tf.matmul(encoding_utterance, W)
        # Batch multiply contexts and utterances (batch_matmul only works with 3-d tensors)
        generated_context = tf.expand_dims(generated_context, 2)
        encoding_context = tf.expand_dims(encoding_context, 2)
        scores = tf.batch_matmul(generated_context, encoding_context, True) + b
        # Go from [15,1,1] to [15,1]: We want a vector of 15 scores
        scores = tf.squeeze(scores, [2])
        # Convert scores into probabilities
        probs = tf.sigmoid(scores)

        # Calculate loss
        loss = tf.contrib.losses.logistic(scores, tf.expand_dims(y, 1))
        tf.scalar_summary("mean_loss", tf.reduce_mean(loss))

    return [probs, loss]


def predict_rnn_batch(contexts, utterances, n=1):
    num_contexts = len(contexts)
    num_records = np.multiply(*utterances.shape)
    input_vectors = []
    for context, utterance_list in zip(contexts, utterances):
        cvec = np.array(list(vocab_processor.transform([context])))
        for u in utterance_list:
            uvec = np.array(list(vocab_processor.transform([u])))
            stacked = np.stack([cvec, uvec], axis=1)
            input_vectors.append(stacked)
    batch = np.vstack(input_vectors)
    result = classifier.predict_proba(batch)[:, 0]
    result = np.split(result, num_contexts)
    return np.argsort(result, axis=1)[:, ::-1]


def evaluate_rnn_predictor(df):
    y_test = np.zeros(len(df))
    y = predict_rnn_batch(df.Context, df.iloc[:, 1:].values)
    for n in [1, 2, 5, 10]:
        print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y, y_test, n)))


class ValidationMonitor(tf.contrib.learn.monitors.BaseMonitor):
    def __init__(self, print_steps=100, early_stopping_rounds=None, verbose=1, val_steps=1000):
        super(ValidationMonitor, self).__init__(
            print_steps=print_steps,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose)
        self.val_steps = val_steps

    def _modify_summary_string(self):
        if self.steps % self.val_steps == 0:
            evaluate_rnn_predictor(validation_df)


def learning_rate_decay_func(global_step):
    return tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps=FLAGS.learning_rate_decay_every,
        decay_rate=FLAGS.learning_rate_decay_rate,
        staircase=True)

classifier = tf.contrib.learn.TensorFlowEstimator(
    model_fn=rnn_encoder_model,
    n_classes=1,
    continue_training=True,
    steps=FLAGS.num_steps,
    learning_rate=learning_rate_decay_func,
    optimizer=FLAGS.optimizer,
    batch_size=FLAGS.batch_size)

monitor = ValidationMonitor(print_steps=100, val_steps=1000)
classifier.fit(X_train, y_train, logdir='./tmp/tf/dual_lstm_chatbot/', monitor=monitor)
