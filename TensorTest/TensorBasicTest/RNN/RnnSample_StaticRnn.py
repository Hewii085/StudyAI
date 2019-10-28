import tensorflow as tf

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None,n_inputs])
X1 = tf.placeholder(tf.float32,[None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)

Y0, Y1 = output_seqs
#Y0 : 각 타임스텝에서의 출력 텐서를 담고 있는 파이썬 리스트
#Y1 : 네트워크의 최종상태 기본적인 셀을 사용 할떄는 최종 상태가 마지막 출력가 동일

init = tf.global_variables_initializer()
X0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
X1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]])

with tf.Session() as sess:
    init
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1:X1_batch})

print(Y0_val)
print(Y1_val)


