import tensorflow as tf

TEXT_DIMENSION = 160


def get_feature_columns(mode):
  feature_columns = []

  # Context
  context_ids = tf.contrib.layers.real_valued_column(
      column_name="context", dimension=TEXT_DIMENSION, dtype=tf.int64)
  feature_columns.append(context_ids)

  # Context Length
  context_lens = tf.contrib.layers.real_valued_column(
      column_name="context_len", dimension=1, dtype=tf.int64)
  feature_columns.append(context_lens)

  # Utterance
  utterance_ids = tf.contrib.layers.real_valued_column(
      column_name="utterance", dimension=TEXT_DIMENSION, dtype=tf.int64)
  feature_columns.append(utterance_ids)

  # Utterance Length
  utterance_lens = tf.contrib.layers.real_valued_column(
      column_name="utterance_len", dimension=1, dtype=tf.int64)
  feature_columns.append(utterance_lens)

  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    labels = tf.contrib.layers.real_valued_column(
      column_name="label", dimension=1, dtype=tf.int64)
    feature_columns.append(labels)
  else:
    for i in range(9):
      # Distractor
      distractor_ids = tf.contrib.layers.real_valued_column(
          column_name="distractor_{}".format(i),
          dimension=TEXT_DIMENSION,
          dtype=tf.int64)
      feature_columns.append(distractor_ids)

      # Distractor Length
      distractor_len = tf.contrib.layers.real_valued_column(
          column_name="distractor_{}_len".format(i),
          dimension=1,
          dtype=tf.int64)
      feature_columns.append(distractor_len)

  return set(feature_columns)


def create_input_fn(mode, input_file, batch_size, num_epochs=None):
  def input_fn():
    features = tf.contrib.layers.create_feature_spec_for_parsing(get_feature_columns(mode))
    feature_map = tf.contrib.learn.io.read_batch_record_features(
        file_pattern=input_file,
        batch_size=batch_size,
        features=features,
        randomize_input=True,
        num_epochs=num_epochs,
        queue_capacity=batch_size * 3)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      target = feature_map.pop("label")
    else:
      # In evaluation we have 10 classes (utterances).
      # The first one is always the correct one
      target = tf.zeros([batch_size, 1], dtype=tf.int64)
    return feature_map, target

  return input_fn
