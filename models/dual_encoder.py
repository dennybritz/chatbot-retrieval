import tensorflow as tf
import numpy as np
from models import helpers

FLAGS = tf.flags.FLAGS

def get_embeddings(hparams):
  if hparams.glove_path and hparams.vocab_path:
    tf.logging.info("Loading Glove embeddings...")
    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
  else:
    tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

  return tf.get_variable(
    "word_embeddings",
    shape=[hparams.vocab_size, hparams.embedding_dim],
    initializer=initializer)


def dual_encoder_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):

  embeddings_W = get_embeddings(hparams)

  with tf.device(embeddings_W.device):
    context_embedded = tf.nn.embedding_lookup(
        embeddings_W, context, name="embed_context")
    utterance_embedded = tf.nn.embedding_lookup(
        embeddings_W, utterance, name="embed_utterance")

  with tf.variable_scope("rnn") as vs:
    cell = tf.nn.rnn_cell.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        state_is_tuple=True)
    # Apply dropout during training
    is_training = tf.convert_to_tensor(mode == tf.contrib.learn.ModeKeys.TRAIN)
    dropout_keep_prob = tf.cond(
        is_training,
        lambda: tf.convert_to_tensor(hparams.dropout_keep_prob),
        lambda: tf.convert_to_tensor(1.0))
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell,
        output_keep_prob=dropout_keep_prob)

    # Run the utterance and context through the RNN
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        cell,
        tf.concat(0, [context_embedded, utterance_embedded]),
        sequence_length=tf.concat(0, [context_len, utterance_len]),
        dtype=tf.float32)
    encoding_context, encoding_utterance = tf.split(0, 2, rnn_states.h)

  with tf.variable_scope("prediction") as vs:
    # Prediction parameters
    W = tf.get_variable(
        "W",
        shape=[encoding_context.get_shape()[1],encoding_utterance.get_shape()[1]],
        initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", [1])
    # Generate a new context
    generated_context = tf.matmul(encoding_utterance, W)
    generated_context = tf.expand_dims(generated_context, 2)
    encoding_context = tf.expand_dims(encoding_context, 2)
    # Compare generated context with actual context
    logits = tf.batch_matmul(generated_context, encoding_context, True) + b
    logits = tf.squeeze(logits, [2])
    probs = tf.sigmoid(logits)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits,
        tf.to_float(targets))

  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
