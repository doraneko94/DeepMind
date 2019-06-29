import tensorflow as tf
import numpy as np
from animal import animalEnv, isvalid
import networkx as nx
import math
import time
import pickle

start = time.time()

gpuConfig = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
    device_count={'GPU': 0})

input_height = 9
input_width = 7
input_channels = 7
conv_n_maps = [64, 64]
conv_kernel_sizes = [(3, 3), (3, 3)]
conv_strides = [1, 1]
conv_paddings = ["SAME"] * 2
conv_activations = [tf.nn.relu] * 2

v_conv_n_map = 1
v_conv_kernel_size = 1
v_conv_stride = 1
v_conv_padding = "SAME"
v_conv_activation = tf.nn.relu
v_n_hidden = 64
v_hidden_activation = tf.nn.relu
v_n_output = 1

p_conv_n_maps = [64, 32]
p_conv_kernel_sizes = [(3, 3), (3, 3)]
p_conv_strides = [1, 1]
p_conv_paddings = ["SAME"] * 2
p_conv_activations = [tf.nn.relu] * 2

learning_rate = 0.2
momentum = 0.95
initializer = tf.variance_scaling_initializer()

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

def softmax(X):
    shape = tf.shape(X)
    X_exp = tf.exp(X)
    X_sum = tf.identity(X_exp)
    for i in [3, 2, 1]:
        X_sum = tf.reduce_sum(X_sum, axis=i)
    X_soft = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(X_sum, 1), 2) ,3), [1, shape[1], shape[2], shape[3]])
    return X_soft

with tf.variable_scope("alpha") as scope:
    conv0 = tf.layers.conv2d(X_state, filters=conv_n_maps[0], kernel_size=conv_kernel_sizes[0], strides=conv_strides[0],
                             padding=conv_paddings[0], activation=conv_activations[0], kernel_initializer=initializer)
    conv1 = tf.layers.conv2d(conv0, filters=conv_n_maps[1], kernel_size=conv_kernel_sizes[1], strides=conv_strides[1],
                             padding=conv_paddings[1], activation=conv_activations[1], kernel_initializer=initializer)                     
    v_conv = tf.layers.conv2d(conv1, filters=v_conv_n_map, kernel_size=v_conv_kernel_size, strides=v_conv_stride,
                              padding=v_conv_padding, activation=v_conv_activation, kernel_initializer=initializer)
    v_conv_flat = tf.reshape(v_conv, shape=[-1, input_height * input_width])
    v_hidden = tf.layers.dense(v_conv_flat, v_n_hidden, activation=v_hidden_activation, kernel_initializer=initializer)
    p_conv0 = tf.layers.conv2d(conv1, filters=p_conv_n_maps[0], kernel_size=p_conv_kernel_sizes[0], strides=p_conv_strides[0],
                               padding=p_conv_paddings[0], activation=p_conv_activations[0], kernel_initializer=initializer)
    p_conv1 = tf.layers.conv2d(p_conv0, filters=p_conv_n_maps[1], kernel_size=p_conv_kernel_sizes[1], strides=p_conv_strides[1],
                               padding=p_conv_paddings[1], activation=p_conv_activations[1], kernel_initializer=initializer)

    p = softmax(p_conv1)
    v = tf.layers.dense(v_hidden, v_n_output, activation=tf.nn.tanh, kernel_initializer=initializer)

learning_rate = 0.2
weight_decay = 1e-4

with tf.variable_scope("train"):
    z = tf.placeholder(tf.float32, shape=[None, 1])
    pi = tf.placeholder(tf.float32, shape=[None, input_height, input_width, 32])

    error = tf.square(z - v)
    p_log = tf.log(p)
    entropy = tf.multiply(pi, p_log)
    entropy_sum = tf.identity(entropy)
    for i in [3, 2, 1]:
        entropy_sum = tf.reduce_sum(entropy_sum, axis=i)
    loss = tf.reduce_mean(error + entropy_sum)
    w_lst = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="alpha")
    for w in w_lst:
        loss += weight_decay * tf.nn.l2_loss(w)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    x = np.random.rand(2, input_height, input_width, input_channels)
    p = np.random.rand(2, input_height, input_width, 32)
    p[0] = p[0] / p[0].sum()
    p[1] = p[1] / p[1].sum()
    v = np.random.rand(2, 1)
    print(loss.eval(feed_dict={X_state: x, pi: p, z: v}))
    print(entropy_sum.eval(feed_dict={X_state: x, pi: p, z: v}))
    print(error.eval(feed_dict={X_state: x, pi: p, z: v}))