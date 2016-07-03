import os
import time
import itertools
import sys
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model

tf.flags.DEFINE_string("test_file", "./data/test.tfrecords", "Path of test data in TFRecords format")
tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("test_batch_size", 16, "Batch size for testing")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

tf.logging.set_verbosity(FLAGS.loglevel)

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=tf.contrib.learn.RunConfig())

  input_fn_test = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[FLAGS.test_file],
    batch_size=FLAGS.test_batch_size,
    num_epochs=1)

  eval_metrics = udc_metrics.create_evaluation_metrics()
  estimator.evaluate(input_fn=input_fn_test, steps=None, metrics=eval_metrics)
