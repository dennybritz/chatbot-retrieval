import tensorflow as tf

def get_features(mode):
  context_features = dict()
  context_features["context_len"] = tf.FixedLenFeature([], dtype=tf.int64)
  context_features["utterance_len"] = tf.FixedLenFeature([], dtype=tf.int64)
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
     context_features["label"] = tf.FixedLenFeature([], dtype=tf.int64)
  else:
    for i in range(9):
      context_features["distractor_{}_len".format(i)] = tf.FixedLenFeature([], dtype=tf.int64)

  sequence_features = dict()
  sequence_features["context"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
  sequence_features["utterance"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
  if mode != tf.contrib.learn.ModeKeys.TRAIN:
    for i in range(9):
      sequence_features["distractor_{}".format(i)] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

  return [context_features, sequence_features]


def create_input_fn(mode, input_file, batch_size, num_epochs=None):
  def input_fn():
    # Get feature columns based on current mode (train/test)
    context_features, sequence_features = get_features(mode)
    # Read an example
    file_queue = tf.train.string_input_producer(input_file, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    seq_key, serialized_example = reader.read(file_queue)
    # Decode the SequenceExample protocol buffer
    context, sequence = tf.parse_single_sequence_example(
      serialized_example,
      context_features=context_features,
      sequence_features=sequence_features,
      example_name="udc_example_{}".format(mode),
    )

    # Merge all features into a single dictionary and batch them
    merged_features = {**context, **sequence}    

    # Get the training labels
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      target = tf.expand_dims(merged_features.pop("label"), 0)
    else:
      target = tf.zeros([1], dtype=tf.int64)

    merged_features["target"] = target
    batched_features = tf.train.batch(
      tensors=merged_features,
      batch_size=batch_size,
      capacity=batch_size * 10,
      dynamic_pad=True)

    return batched_features, batched_features["target"]

  return input_fn
