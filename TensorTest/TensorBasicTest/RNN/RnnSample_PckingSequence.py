import tensorflow as tf
import numpy as np


n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32,[None, n_steps,n_inputs])
X_seqs = tf.unstack(tf.transpose(X,perm=[1,0,2]))
#첫번째 차원을 따라 파이썬 리스트를 추출

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])
init = tf.global_variables_initializer()

X_batch = np.array([ [[0,1,2],[9,8,7]],
                     [[3,4,5],[0,0,0]],
                     [[6,7,8],[6,5,4]],
                     [[9,0,1],[3,2,1]] ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X:X_batch})

print(outputs_val)
print(np.transpose(outputs_val, axes=[1,0,2])[1])