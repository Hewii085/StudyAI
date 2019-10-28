import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx,n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
Y = tf.placeholder(tf.int32,[None])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits,Y,1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

(X_train,Y_train),(X_test,Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)
X_valid, X_train= X_train[:5000], X_train[5000:]
Y_valid, Y_train = Y_train[:5000], Y_train[5000:]

X_test = X_test.reshape((-1, n_steps,n_inputs))

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train,Y_train,batch_size):
            X_batch = X_batch.reshape((-1,n_steps,n_inputs))
            sess.run(train_op, feed_dict={X:X_batch, Y:y_batch})

        acc_batch = accuracy.eval(feed_dict={X:X_batch, Y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:X_test, Y:Y_test})
        print(epoch,"Last batch accuracy : ", acc_batch, "Test accuracy : ",acc_test)
    while True:
        r = random.randint(0,len(X_valid))
        evalImg = X_valid[r].reshape(1,28,28).astype(np.float32)
        rslt = sess.run(logits,feed_dict={X:evalImg})    
        print("predict : ",sess.run(tf.argmax(rslt,1)) , "Label : ",Y_valid[r])
    #Evaluation Data로 테스트 해보고 싶은데?