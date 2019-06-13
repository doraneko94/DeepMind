import tensorflow as tf
import numpy as np
from functools import partial

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

training = tf.placeholder_with_default(False, shape=(), name="training")
my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = tf.layers.batch_normalization(hidden1)
    bn1_act = tf.nn.elu(bn1)
    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = tf.layers.batch_normalization(hidden2)
    bn2_act = tf.nn.elu(bn2)
    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

threthold = 1.0

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threthold, threthold), var) for grad, var in grads_and_vars]
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("mnist.pkl", "rb") as f:
    mnist = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], random_state=42, test_size=0.33)
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, random_state=42, test_size=0.33)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)

n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_eval, y: y_eval})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy", acc_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = X_test
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print(y_pred[:10])
print(y_test[:10])