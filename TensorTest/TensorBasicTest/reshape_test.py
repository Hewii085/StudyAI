import tensorflow as tf
import numpy as np


def case_three():
        inputShape = np.arange(13*13*30)
        inputShape = np.reshape(inputShape,[13,13,5,6]).astype(np.float32)

        print(inputShape)
        x = tf.placeholder(tf.float32, [13,13,5,6])
        sig = tf.sigmoid(x) + 1

        with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                rslt = sess.run(sig, feed_dict={x:inputShape})
                print(rslt)


def case_two():
        inputShape = np.arange(13*13*30)
        inputShape = np.reshape(inputShape,[13,13,30])

        x = tf.placeholder(tf.int32,[13,13,30])
        rslt = tf.reshape(x,[13,13,5,6])
        
        with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                evalRslt = sess.run(rslt,feed_dict={x:inputShape})

        print(evalRslt)

def case_one():
        x = tf.placeholder(tf.float32,[-1,13,13,30])
        rslt = tf.reshape(x,[-1,13,13,5,6])
        sess = tf.InteractiveSession()
        print(rslt.eval())

case_three()