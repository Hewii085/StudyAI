import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random

epoch = 10
batchSize = 100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
num_itr = int(mnist.train.num_examples / batchSize)


def activation(input):
    return tf.nn.leaky_relu(input,alpha=0.1)

def train():
    x_input = tf.placeholder(tf.float32,[None, 784])
    y_output = tf.placeholder(tf.float32,[None, 10])

    layer1 = tf.Variable(tf.random_normal([784,256]))
    bias1 = tf.Variable(tf.random_normal([256]))
    hypothesis = tf.matmul(x_input,layer1)*bias1
    # hypothesis = tf.nn.leaky_relu(hypothesis,alpha=0.1)

    layer2 = tf.Variable(tf.random_normal([256,10]))
    bias2 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(hypothesis,layer2)*bias2
    # hypothesis = tf.nn.leaky_relu(hypothesis,alpha=0.1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,labels=tf.stop_gradient(y_output)))
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(epoch):
            avgCost = 0

            for itr in range(num_itr):
                xs,ys = mnist.train.next_batch(batchSize)
                _, costVal = sess.run([train,cost], feed_dict={x_input:xs, y_output:ys})
                avgCost += costVal / num_itr

            print("Cost :", avgCost)

        while True:
            r = random.randint(0, mnist.test.num_examples - 1)
            img = mnist.test.images[r].reshape(1,784)
            accuracy = sess.run(hypothesis,feed_dict={x_input:img})
            #print("Accuracy : ",accuracy)
            print("Num : ",sess.run(tf.argmax(accuracy,1)))

            #plt.imshow(img.reshape(28,28), cmap="Greys")
            #plt.show()
            time.sleep(3)
            print("Finished Learning")

    return hypothesis


logit = train()
# eval(logit)