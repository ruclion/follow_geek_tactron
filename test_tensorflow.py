import tensorflow as tf
import numpy as np

init_a = 3 * np.ones((2, 2), dtype=np.float32)

a = tf.get_variable(name='a', dtype=tf.float32, initializer=init_a)
# a = tf.constant(3, dtype=tf.float32, shape=(2, 2))
# print(a.get_shape()[-1])
b = tf.constant(5, dtype=tf.float32, shape=(2, 2))

w = tf.get_variable('w', shape=(1), dtype=tf.float32, initializer=tf.constant_initializer(1.0))
b_pre = w * a
loss = tf.reduce_mean(tf.abs(b_pre - b))

opt = tf.train.AdamOptimizer(0.1)
grads_and_vars = opt.compute_gradients(loss)

session = tf.Session()





train_upd = opt.apply_gradients(grads_and_vars)
session.run(tf.initialize_all_variables())
print(session.run(a))
print(session.run(grads_and_vars))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

session.run(train_upd)

print(session.run(a))
print(session.run(w))

