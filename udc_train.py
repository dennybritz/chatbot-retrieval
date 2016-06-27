import os
import time
import itertools
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model

tf.flags.DEFINE_string("input_dir", "./data", "")
tf.flags.DEFINE_string("model_dir", None, "")
tf.flags.DEFINE_integer("loglevel", 20, "Log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of Training Epochs")
tf.flags.DEFINE_integer("eval_every", 1000, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("num_eval_steps", 100, "Number of Eval Steps")
FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())
if FLAGS.model_dir:
  MODEL_DIR = FLAGS.model_dir
else:
  MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_argv):
  hparams = udc_hparams.create_hparams()

  model_fn = udc_model.create_model_fn_for_recall(
    hparams,
    model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig())

  input_fn_eval = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_file=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size)

  eval_metrics = udc_metrics.create_evaluation_metrics(hparams)

  # TODO: Currently the validation monitor doesn't support metrics.
  # It's on the master branch so we need to wait for next TF release
  # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
  #   input_fn=input_fn_eval,
  #   eval_steps=FLAGS.num_eval_steps,
  #   every_n_steps=FLAGS.eval_every,
  #   metrics=udc_metrics.create_evaluation_metrics(hparams)
  # )

  while True:
    input_fn_train = udc_inputs.create_input_fn(
      mode=tf.contrib.learn.ModeKeys.TRAIN,
      input_file=[TRAIN_FILE],
      batch_size=hparams.batch_size,
      num_epochs=1)    
    estimator.fit(input_fn=input_fn_train, steps=None)
    estimator.evaluate(input_fn=input_fn_eval, steps=FLAGS.num_eval_steps, metrics=eval_metrics)


if __name__ == "__main__":
  tf.app.run()
