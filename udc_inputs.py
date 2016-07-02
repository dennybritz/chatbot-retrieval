import tensorflow as tf

def get_features(mode):
  context_features = dict()
  context_features["context_len"] = tf.FixedLenFeature([], dtype=tf.int64)
  context_features["utterance_len"] = tf.FixedLenFeature([], dtype=tf.int64)
  # Only the training data has a label
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
     context_features["label"] = tf.FixedLenFeature([], dtype=tf.int64)
  else:
    # Only test/validation data has distractor lengths
    for i in range(9):
      context_features["distractor_{}_len".format(i)] = tf.FixedLenFeature([], dtype=tf.int64)

  sequence_features = dict()
  sequence_features["context"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
  sequence_features["utterance"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
  # Only test/validation data has distractor
  if mode != tf.contrib.learn.ModeKeys.TRAIN:
    for i in range(9):
      sequence_features["distractor_{}".format(i)] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

  return [context_features, sequence_features]


def trim_pad_tensor(t, length):
  """
  Trims or pads a vector to the specified length.
  """
  return tf.cond(
    tf.greater_equal(tf.size(t), length),
    lambda: tf.slice(t, [0], [length]),
    lambda: tf.pad(
      t,
      tf.convert_to_tensor([[0,1]]) * (tf.constant(length) - tf.size(t)))
  )

def create_input_fn(mode, hparams, input_files, batch_size, num_epochs=None):
  def input_fn():

    # Get feature columns based on current mode (train/test)
    context_features, sequence_features = get_features(mode)

    # Read an example
    file_queue = tf.train.string_input_producer(input_files, num_epochs=num_epochs, name="{}_inputs".format(mode))
    reader = tf.TFRecordReader()
    seq_key, serialized_example = reader.read(file_queue)

    # Decode the SequenceExample protocol buffer
    context, sequence = tf.parse_single_sequence_example(
      serialized_example,
      context_features=context_features,
      sequence_features=sequence_features,
      example_name="udc_example_{}".format(mode),
    )

    # This is an ugly hack because of a current bug in tf.learn
    # During evaluation TF tries to restore the epoch variable which isn't defined during training
    # So we define the variable manually here
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      tf.get_variable("eval_inputs/limit_epochs/epochs", initializer=tf.constant(0, dtype=tf.int64))

    # Merge all features into a single dictionary and batch them
    merged_features = context.copy()
    merged_features.update(sequence)

    # Trim or pad utterances to the same length
    max_len = max([hparams.max_utterance_len, hparams.max_context_len])
    merged_features["utterance"] = trim_pad_tensor(
      merged_features["utterance"],
      max_len)
    if mode == tf.contrib.learn.ModeKeys.EVAL:
      for i in range(9):
        key = "distractor_{}".format(i)
        merged_features[key] = trim_pad_tensor(
          merged_features[key], max_len)

    # Trim or pad contexts to the same length
    merged_features["context"] = trim_pad_tensor(
      merged_features["context"], max_len)

    # Get the training labels
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      target = tf.expand_dims(merged_features.pop("label"), 0)
    else:
      target = tf.zeros([1], dtype=tf.int64)

    # Batch the features
    merged_features["target"] = target
    batched_features = tf.train.batch(
      tensors=merged_features,
      batch_size=batch_size,
      capacity=100 + batch_size * 10,
      dynamic_pad=True)
    labels = batched_features["target"]

    return batched_features, labels

  return input_fn
