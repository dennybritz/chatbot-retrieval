import tensorflow as tf

TEXT_FEATURE_SIZE = 160

def get_feature_columns(mode):
  feature_columns = []

  feature_columns.append(tf.contrib.layers.real_valued_column(
    column_name="context", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="context_len", dimension=1, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="utterance", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="utterance_len", dimension=1, dtype=tf.int64))

  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    # During training we have a label feature
    feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="label", dimension=1, dtype=tf.int64))

  if mode == tf.contrib.learn.ModeKeys.EVAL:
    # During evaluation we have distractors
    for i in range(9):
      feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="distractor_{}".format(i), dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
      feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))

  return set(feature_columns)


def create_input_fn(mode, input_files, batch_size, num_epochs):
  def input_fn():
    features = tf.contrib.layers.create_feature_spec_for_parsing(
        get_feature_columns(mode))

    feature_map = tf.contrib.learn.io.read_batch_features(
        file_pattern=input_files,
        batch_size=batch_size,
        features=features,
        reader=tf.TFRecordReader,
        randomize_input=True,
        num_epochs=num_epochs,
        queue_capacity=200000 + batch_size * 10,
        name="read_batch_features_{}".format(mode))

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      target = feature_map.pop("label")
    else:
      # In evaluation we have 10 classes (utterances).
      # The first one (index 0) is always the correct one
      target = tf.zeros([batch_size, 1], dtype=tf.int64)
    return feature_map, target
  return input_fn
