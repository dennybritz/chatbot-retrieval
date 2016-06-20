#! /usr/bin/env python

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.contrib import learn
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import dtypes
from helpers import load_glove_vectors, evaluate_recall

logging.set_verbosity(10)

# Learning Parameters
tf.flags.DEFINE_integer("num_steps", 1000000, "Number of training steps")
tf.flags.DEFINE_integer("batch_size", 256, "Batch size")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")
tf.flags.DEFINE_boolean("use_glove", False, "Use pre-trained glove vectors")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout Keep Probability")
tf.flags.DEFINE_float("learning_rate_decay_rate", 0.1, "Learning Rate Decay Factor")
tf.flags.DEFINE_integer("learning_rate_decay_every", 3000, "Decay after this many steps")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer (Adam, Adagrad or SGD)")

# Model Parameters
tf.flags.DEFINE_integer("max_content_length", 80, "Maximum length of context in words")
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
validation_df = pd.read_csv(os.path.join(FLAGS.data_dir, "valid.csv"))
test_df = pd.read_csv(os.path.join(FLAGS.data_dir, "test.csv"))
y_test = np.zeros(len(test_df))


# Preprocessing
# ==================================================
# Create vocabulary mapping
all_sentences = np.append(train_df.Context, train_df.Utterance)
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_CONTEXT_LENGTH, min_frequency=5)
vocab_processor.fit(all_sentences)

# Transform contexts and utterances
X_train_context = np.array(list(vocab_processor.transform(train_df.Context)))
X_train_utterance = np.array(list(vocab_processor.transform(train_df.Utterance)))

# Generate training tensor
X_train = np.stack([X_train_context, X_train_utterance], axis=1)
y_train = train_df.Label

n_words = len(vocab_processor.vocabulary_)
print("Total words: {}".format(n_words))

# Convert to tf.Example Proto
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "tmp", timestamp))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("Writing to {}".format(out_dir))
tfrecords_filename = os.path.join(out_dir, "train.tfrecords")
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for index in range(len(X_train)):
    context = tf.train.Feature(int64_list=tf.train.Int64List(value=X_train[index][0].tolist()))
    utterance = tf.train.Feature(int64_list=tf.train.Int64List(value=X_train[index][1].tolist()))
    label = tf.train.Feature(int64_list=tf.train.Int64List(value=[y_train[index].item()]))
    example = tf.train.Example(features=tf.train.Features(feature={
        'context': context,
        'utterance': utterance,
        'label': label}))
    writer.write(example.SerializeToString())
writer.close()


# Input Examples
def get_input():
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=100)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    decoded = tf.parse_single_example(
        serialized_example,
        features={
            'context': tf.FixedLenFeature([MAX_CONTEXT_LENGTH], tf.int64),
            'utterance': tf.FixedLenFeature([MAX_CONTEXT_LENGTH], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    example_X = tf.concat(0, [tf.expand_dims(decoded['context'], 0), tf.expand_dims(decoded['utterance'], 0)])
    example_y = decoded['label']
    return tf.train.batch([example_X, example_y], FLAGS.batch_size)


# Load glove vectors
# ==================================================
if FLAGS.use_glove:
    vocab_set = set(vocab_processor.vocabulary_._mapping.keys())
    glove_vectors, glove_dict = load_glove_vectors(os.path.join(FLAGS.data_dir, "glove.840B.300d.txt"), vocab_set)


# Build initial word embeddings
# ==================================================
initial_embeddings = np.random.uniform(-0.25, 0.25, (n_words, EMBEDDING_DIM)).astype("float32")
if FLAGS.use_glove:
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
    sequence_lengths = zero_positions + (position_mask * max_length)
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
        word_vectors_context = learn.ops.embedding_lookup(embeddings, context)
        # Embed the utterance
        word_vectors_utterance = learn.ops.embedding_lookup(embeddings, utterance_truncated)

    # Run context and utterance through the same RNN
    with tf.variable_scope("shared_rnn_params") as vs:
        cell = tf.nn.rnn_cell.LSTMCell(RNN_DIM, forget_bias=2.0)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.dropout_keep_prob)
        context_outputs, context_state = tf.nn.dynamic_rnn(
            cell, word_vectors_context, dtype=dtypes.float32, sequence_length=context_seq_length)
        encoding_context = tf.slice(context_state, [0, cell.output_size], [-1, -1])
        vs.reuse_variables()
        utterance_outputs, utterance_state = tf.nn.dynamic_rnn(
            cell, word_vectors_utterance, dtype=dtypes.float32, sequence_length=utterance_seq_length)
        encoding_utterance = tf.slice(utterance_state, [0, cell.output_size], [-1, -1])

    with tf.variable_scope("prediction") as vs:
        W = tf.get_variable("W",
                            shape=[encoding_context.get_shape()[1], encoding_utterance.get_shape()[1]],
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable("b", [1])

        # We can interpret this is a "Generated context"
        generated_context = tf.matmul(encoding_utterance, W)
        # return learn.models.logistic_regression(generated_context, tf.expand_dims(y, 1))
        # Batch multiply contexts and utterances (batch_matmul only works with 3-d tensors)
        generated_context = tf.expand_dims(generated_context, 2)
        encoding_context = tf.expand_dims(encoding_context, 2)
        logits = tf.batch_matmul(generated_context, encoding_context, True) + b
        logits = tf.squeeze(logits)
        probs = tf.sigmoid(logits)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(y))

    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    train_op = tf.contrib.layers.optimize_loss(
      mean_loss, tf.contrib.framework.get_global_step(),
      optimizer=FLAGS.optimizer,
      learning_rate=FLAGS.learning_rate,
      moving_average_decay=None)

    return {'class': tf.argmax(probs, 1), 'prob': probs}, mean_loss, train_op


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


class ValidationMonitor(tf.contrib.learn.monitors.EveryN):
    def __init__(self, every_n_steps=100, early_stopping_rounds=None, verbose=1, val_steps=1000):
        super(ValidationMonitor, self).__init__(
            every_n_steps=every_n_steps,
            first_n_steps=1)

    def every_n_step_end(self, step, outputs):
        super(ValidationMonitor, self).every_n_step_end(step, outputs)
        evaluate_rnn_predictor(validation_df)


def learning_rate_decay_func(global_step):
    return tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps=FLAGS.learning_rate_decay_every,
        decay_rate=FLAGS.learning_rate_decay_rate,
        staircase=True)

classifier = tf.contrib.learn.Estimator(
    model_fn=rnn_encoder_model,
    model_dir='./tmp/tf/dual_lstm_chatbot/')

# monitor = ValidationMonitor(every_n_steps=100)
monitor = learn.monitors.ValidationMonitor(X_train[:500], y_train[:500])
classifier.fit(input_fn=get_input, steps=None, monitors=[monitor])
