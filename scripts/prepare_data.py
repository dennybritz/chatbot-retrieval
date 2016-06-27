import os
import csv
import itertools
import functools
import tensorflow as tf
import numpy as np
import array

tf.flags.DEFINE_integer(
  "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer(
  "max_sentence_len", 160,
  "Maximum Sentence length. All text will be trimmed to this length.")

tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("./data"),
  "Output directory for TFrEcord files (default = './data')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.csv")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")

def create_csv_iter(filename):
  """
  Returns an iterator over a CSV file. Skips the header.
  """
  with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for row in reader:
      yield row


def create_vocab(input_iter, min_frequency):
  """
  Creates and returns a VocabularyProcessor object with the vocabulary
  for the input iterator.
  """
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      FLAGS.max_sentence_len,
      min_frequency=min_frequency,
      tokenizer_fn=lambda iterator: (x.split(" ") for x in iterator))
  vocab_processor.fit(input_iter)
  return vocab_processor


def transform_sentence(sequence, vocab_processor):
  """
  Maps a single sentence into the integer vocabulary. Returns a python array.
  """
  return list(vocab_processor.transform([sequence]))[0].tolist()


def create_example_train(row, vocab):
  """
  Creates a training example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, utterance, label = row
  label = int(float(label))

  # Context
  context_feature = tf.train.Feature(int64_list=tf.train.Int64List(
      value=transform_sentence(context, vocab)))
  context_text_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[context.encode()]))

  # Context Length
  context_len = len(next(vocab._tokenizer([context])))
  context_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[context_len]))

  # Utterance
  utterance_feature = tf.train.Feature(int64_list=tf.train.Int64List(
      value=transform_sentence(utterance, vocab)))
  utterance_text_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[utterance.encode()]))

  # Utterance Length
  utterance_len = len(next(vocab._tokenizer([utterance])))
  utterance_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[utterance_len]))

  label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
  example = tf.train.Example(features=tf.train.Features(feature={
      "context": context_feature,
      # "context_text": context_text_feature,
      "context_len": context_len_feature,
      "utterance": utterance_feature,
      # "utterance_text": utterance_text_feature,
      "utterance_len": utterance_len_feature,
      "label": label_feature
  }))
  return example


def create_example_test(row, vocab):
  """
  Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """  
  context, utterance = row[:2]
  distractors = row[2:]
  context_feature = tf.train.Feature(int64_list=tf.train.Int64List(
      value=transform_sentence(context, vocab)))
  context_text_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[context.encode()]))

  # Context Length
  context_len = len(next(vocab._tokenizer([context])))
  context_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[context_len]))

  utterance_feature = tf.train.Feature(int64_list=tf.train.Int64List(
      value=transform_sentence(utterance, vocab)))
  utterance_text_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[utterance.encode()]))

  # Utterance Length
  utterance_len = len(next(vocab._tokenizer([utterance])))
  utterance_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[utterance_len]))

  feature_map = {
      "context": context_feature,
      "context_len": context_len_feature,
      # "context_text": context_text_feature,
      "utterance": utterance_feature,
      # "utterance_text": utterance_text_feature,
      "utterance_len": utterance_len_feature,
  }
  for i, distractor in enumerate(distractors):
    # Distractor Feature
    distractor_ids = transform_sentence(distractor, vocab)
    feature_map["distractor_{}".format(i)] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=distractor_ids))

    # distractor_text_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[distractor.encode()]))
    # feature_map["distractor_{}_text".format(i)] = distractor_text_feature

    # Distractor Length
    dis_len = len(next(vocab._tokenizer([distractor])))
    dis_len_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[dis_len]))
    feature_map["distractor_{}_len".format(i)] = dis_len_feature

  example = tf.train.Example(features=tf.train.Features(feature=feature_map))
  return example


def create_tfrecords_file(input_filename, output_filename, example_fn):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for i, row in enumerate(create_csv_iter(input_filename)):
    x = example_fn(row)
    writer.write(x.SerializeToString())
  writer.close()
  print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
  """
  Writes the vocabulary to a file, one word per line.
  """
  vocab_size = len(vocab_processor.vocabulary_)
  with open(outfile, "w") as vocabfile:
    for id in range(vocab_size):
      word =  vocab_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
  print("Creating vocabulary...")
  input_iter = create_csv_iter(TRAIN_PATH)
  input_iter = (x[0] + " " + x[1] for x in input_iter)
  vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
  print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

  write_vocabulary(
      vocab,
      os.path.join(FLAGS.output_dir, "vocabulary.txt"))

  # Create validation.tfrecords
  create_tfrecords_file(
      input_filename=VALIDATION_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create test.tfrecords
  create_tfrecords_file(
      input_filename=TEST_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create train.tfrecords
  create_tfrecords_file(
      input_filename=TRAIN_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
      example_fn=functools.partial(create_example_train, vocab=vocab))
