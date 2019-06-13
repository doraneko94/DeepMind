import numpy as np
import tensorflow as tf

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")
theta = tf.get_default_graph().get_tensor_by_name("theta:0")
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")

    best_theta = theta.eval()
print(best_theta)