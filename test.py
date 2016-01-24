import tensorflow as tf
a = tf.constant(5)
b = tf.constant(10)
print(tf.Session().run(a+b))