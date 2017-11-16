import  tensorflow as tf
import numpy as np


state_dim = 8
num_actions = 5

with tf.variable_scope("q_network"):
    W1 = tf.Variable(np.arange(state_dim * 20).reshape(state_dim, 20), dtype=tf.float32, name="W1")
    b1 = tf.Variable(np.arange(20), dtype=tf.float32, name="b1")

    W2 = tf.Variable(np.arange(20 * num_actions).reshape(20, num_actions), dtype=tf.float32, name="W2")
    b2 = tf.Variable(np.arange(num_actions), dtype=tf.float32, name="b2")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")

    print("W1:", sess.run(W1))
    print("b1:", sess.run(b1))

    print("W2:", sess.run(W2))
    print("b2:", sess.run(b2))