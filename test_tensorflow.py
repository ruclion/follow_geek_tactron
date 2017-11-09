import tensorflow as tf

a = tf.constant(3, dtype=tf.float32, shape=(5, 3, 2))
b = tf.constant(5, shape=(1, 3, 2))
d = tf.nn.softmax(a)

c = tf.reshape(b, (-1, 1, 2))


session = tf.Session()
print(d, session.run(d))
