import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape=[2, 3, 4, 5])

def softmax(X):
    shape = tf.shape(X)
    X_exp = tf.exp(X)
    X_sum = tf.identity(X_exp)
    for i in [3, 2, 1]:
        X_sum = tf.reduce_sum(X_sum, axis=i)
    X_soft = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(X_sum, 1), 2) ,3), [1, shape[1], shape[2], shape[3]])
    return X_soft

X_soft = softmax(X)

state = np.array([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                   [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                   [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]],
                  [[[10, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                   [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                   [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]])

with tf.Session() as sess:
    print(X_soft.eval(feed_dict={X: state}))
