import tensorflow as tf
import functools

def create_evaluation_metrics():
  eval_metrics = {}
  for k in [1, 2, 5, 10]:
    eval_metrics["recall_at_%d" % k] = functools.partial(
        tf.contrib.metrics.streaming_sparse_recall_at_k,
        k=k)
  return eval_metrics