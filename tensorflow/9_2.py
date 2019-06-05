import tensorflow as tf

x = tf.Variable(3)
y = tf.Variable(4)
f = x*x*y + y + 2

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)
