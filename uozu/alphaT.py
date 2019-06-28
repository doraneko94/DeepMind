import tensorflow as tf
import numpy as np
from ox import oxEnv, isvalid
import networkx as nx
import math
import time
import pickle

start = time.time()

gpuConfig = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
    device_count={'GPU': 0})

input_height = 3
input_width = 3
input_channels = 7
conv_n_maps = [4, 8]
conv_kernel_sizes = [(2, 2), (1, 1)]
conv_strides = [1, 1]
conv_paddings = ["VALID"] * 2
conv_activation = [tf.nn.relu] * 2

n_hidden_in = 2 * 2 * 8
n_hidden = 16
hidden_activation1 = tf.nn.sigmoid
hidden_activation2 = tf.nn.tanh
n_outputs1 = 9
n_outputs2 = 1
scale = 0.001
initializer = tf.variance_scaling_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale)

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

def alpha(X_state, name):
    prev_layer = X_state

    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
            conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation
            ):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer, kernel_regularizer=regularizer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden1 = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                  activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        hidden2 = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                  activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        hidden3 = tf.layers.dense(hidden1, n_hidden,
                                  activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        p_raw = tf.layers.dense(hidden3, n_outputs1, activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        #p = tf.nn.softmax(p_raw)
        p = p_raw / tf.reduce_sum(p_raw)
        v = tf.layers.dense(hidden2, n_outputs2, activation=hidden_activation2, kernel_initializer=initializer, kernel_regularizer=regularizer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return p, v, trainable_vars_by_name

p_new, v_new, vars_new = alpha(X_state, "alpha/new")

learning_rate = 0.01

with tf.variable_scope("train"):
    z = tf.placeholder(tf.float32, shape=[None, 1])
    pi = tf.placeholder(tf.float32, shape=[None, 9])

    error = tf.square(z - v_new)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi , logits=p_new)
    loss = tf.reduce_mean(error + entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# [1, 0, 1, 0, -1, 0, 0, -1, 0]
state_batch = np.array([[[[ 1,  0,  1],  
                          [ 0,  0,  0], 
                          [ 0,  0,  0]],
                         [[ 0,  0,  0], 
                          [ 0, -1,  0], 
                          [ 0, -1,  0]],
                         [[ 1,  0,  1], 
                          [ 0,  0,  0], 
                          [ 0,  0,  0]],
                         [[ 0,  0,  0], 
                          [ 0,  0,  0], 
                          [ 0, -1,  0]],
                         [[ 1,  0,  0], 
                          [ 0,  0,  0],  
                          [ 0,  0,  0]],
                         [[ 0,  0,  0],  
                          [ 0,  0,  0],  
                          [ 0,  0,  0]],
                         [[ 1,  1,  1],  
                          [ 1,  1,  1],  
                          [ 1,  1,  1]]]])
state_batch = state_batch.reshape((1, 3, 3, 7))
pi_batch = np.array([[0.05, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]])
z_batch = np.array([[1]])

if __name__=="__main__":
    with tf.Session(config=gpuConfig) as sess:
        init.run()
        #print(entropy.eval(feed_dict={X_state: state_batch, z: z_batch, pi: pi_batch}))
        
        for i in range(30001):
            training_op.run(feed_dict={X_state: state_batch, z: z_batch, pi: pi_batch})
            if i % 100 == 0:

                print(v_new.eval(feed_dict={X_state: state_batch}))
                print(p_new.eval(feed_dict={X_state: state_batch}))
        