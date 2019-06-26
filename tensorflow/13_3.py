import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
from matplotlib import pyplot as plt

china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")
convolution = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2, 2], padding="SAME")
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(convolution, feed_dict={X: dataset})

print(channels)
print(output.shape)
plt.imshow(output[0, :, :, 1], cmap="gray")
plt.show()