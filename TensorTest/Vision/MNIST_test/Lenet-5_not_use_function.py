import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data

def pad(x):
    rslt = np.pad(x,((0,0),(2,2),(2,2),(0,0)),'constant')
    return rslt

EPOCHS = 50
BATCH_SIZE = 128

mnist = input_data.read_data_sets("MNIST_data",reshape=False)
x_train, y_train = mnist.train.images, mnist.train.labels
x_train = pad(x_train)

x=tf.placeholder(tf.float32,(None,32,32,1))
y=tf.placeholder(tf.int32,(None))
one_hot_y = tf.one_hot(y,10)

conv1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6],mean=0,stddev=0.01))
conv1_bias = tf.Variable(tf.zeros(6))
conv1 = tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1], padding='VALID') + conv1_bias
conv1 = tf.nn.relu(conv1)
pool_1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

conv2_w = tf.Variable(tf.truncated_normal([5,5,6,16],mean=0,stddev=0.01))
conv2_bias = tf.Variable(tf.zeros(16))
conv2 = tf.nn.conv2d(pool_1,conv2_w,strides=[1,1,1,1],padding='VALID') + conv2_bias
conv2 = tf.nn.relu(conv2)
pool_2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

fc1 = flatten(pool_2)

fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120),mean=0, stddev=0.01))
fc1_b = tf.Variable(tf.zeros(120))
fc1 = tf.matmul(fc1, fc1_w) + fc1_b

fc1 = tf.nn.relu(fc1)

fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84),mean=0,stddev=0.01))
fc2_b = tf.Variable(tf.zeros(84))
fc2 = tf.matmul(fc1,fc2_w)+fc2_b
fc2 = tf.nn.relu(fc2)

fc3_w = tf.Variable(tf.truncated_normal(shape=(84,10),mean=0,stddev=0.01))
fc3_b = tf.Variable(tf.zeros(10))
logits = tf.matmul(fc2,fc3_w)+fc3_b

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_operation = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_examples = len(x_train)

for i in range(EPOCHS):
    avg_cost=0
    for offset in range(0,num_examples,BATCH_SIZE):
        end = offset+BATCH_SIZE
        batch_x, batch_y = x_train[offset:end],y_train[offset:end]
        c=sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                # avg_cost += c/int(num_examples/BATCH_SIZE)
    print('Epoch:', '%04d' % (i + 1))
    
print("Finished Training")


for r in range(11):
    testDigit = mnist.train.images[r]
    testDigit = np.reshape(testDigit,[1,28,28,1]).astype(np.float32)
    testDigit = np.pad(testDigit,((0,0),(2,2),(2,2),(0,0)),'constant')
    rslt = sess.run(tf.argmax(logits,1),feed_dict={x:testDigit})
    print("Prediction:",rslt)
    plt.imshow(testDigit[0,:,:,0],cmap='Greys')
    plt.show()





