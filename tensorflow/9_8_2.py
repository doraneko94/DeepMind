import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32)
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32)
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0))
y_pred = tf.matmul(X, theta)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")

    best_theta = theta.eval()
print(best_theta)