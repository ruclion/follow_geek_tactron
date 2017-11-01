import tensorflow as tf
from tensorflow.python.ops import losses
import math

compute_weighted_loss = losses.losses_impl.compute_weighted_loss

def sparse_categorical_accuracy(y_true, y_pred, mask=1):
    _, max_ind = tf.nn.top_k(y_pred)
    max_ind = tf.cast(tf.squeeze(max_ind), tf.int32)
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    score = tf.cast(tf.equal(y_true, max_ind), tf.float32)
    return compute_weighted_loss(score, mask)

def binary_accuracy(y_true, y_pred, mask=1):
    round_y_pred = tf.round(y_pred)
    right_cnt = tf.cast(tf.equal(y_true, round_y_pred), tf.float32)
    return compute_weighted_loss(right_cnt, mask)

def perplexity(label, logit):
    words = tf.cast(tf.size(label), tf.float32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
    cross_entropy = tf.divide(tf.reduce_sum(cross_entropy), words)
    perplex = tf.pow(2.0, cross_entropy)
    return perplex

