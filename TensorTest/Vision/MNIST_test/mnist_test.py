import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
batch_size =100
mnist_idx  = 20

def forth_test():
    #max pool
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    digit = mnist.train.images[mnist_idx]
    digitReshape = np.reshape(digit,[1,28,28,1])
    x = tf.placeholder(tf.float32,[1,28,28,1])
    L1 = tf.nn.max_pool(digitReshape, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        max_op=session.run(L1,feed_dict={x:digitReshape})
    
    plt.imshow(max_op[0,:,:,0],cmap="Greys")
    plt.show()
      

def third_test():
    #layer 3
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    digit = mnist.train.images[mnist_idx]
    digitReshape = np.reshape(digit,[1,28,28,1])
    x = tf.placeholder(tf.float32,[1,28,28,1])
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(digitReshape,W1,strides=[1,1,1,1],padding="SAME")
    #L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding="SAME")
    #L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    #L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        conv_op=session.run(L3,feed_dict={x:digitReshape})

    for i in range(1,conv_op.shape[3]):
        aa= conv_op[0,:,:,i]
        plt.imshow(aa,cmap="Greys")
        plt.show()

def second_test():
    #layer 1
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    digit = mnist.train.images[mnist_idx]
    digitReshape = np.reshape(digit,[1,28,28,1])
    x = tf.placeholder(tf.float32,[1,28,28,1])
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

    conv = tf.nn.conv2d(digitReshape,W1,strides=[1,1,1,1],padding="SAME")
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        conv_op=session.run(conv,feed_dict={x:digitReshape})

    for i in range(1,conv_op.shape[3]):
        aa= conv_op[0,:,:,i]
        #im = np.reshape(conv_op[0,:,:,:],[28,28]).astype(np.float32)
        plt.imshow(aa,cmap="Greys")
        plt.show()
       

def first_test():
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    digit = mnist.train.images[mnist_idx]
    digitReshape = np.reshape(digit,[1,28,28,1])

    x = tf.placeholder(tf.float32,[1,28,28,1])
    kernel = np.array([[0,0,0],[0,1,0],[0,0,0]]).astype(np.float32)

    kernelReshape= np.reshape(kernel,[3,3,1,1])
    #conv2d input은 [batch, in_height, in_width,in_channel] 로 이루어진다.
    #conv2d kerenel shape 는 [height,with,in_channel, out_channel] 형태로 맞춘다.
    #out_channel -> output filter 개수를 말하는거 같다

    conv = tf.nn.conv2d(digitReshape,kernelReshape,strides=[1,1,1,1],padding="SAME")
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        conv_op=session.run(conv,feed_dict={x:digitReshape})

    im = np.reshape(conv_op,[28,28])
    plt.imshow(im,cmap="Greys")
    plt.show()

if __name__ == '__main__':
    third_test()
# x = tf.placeholder(tf.float32, [1,28, 28,1])
# x_img = tf.reshape(x,[1,28,28,1])
# weight = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))

# conv = tf.nn.conv2d(image,weight,strides=[1,1,1,1], padding='SAME')

# init = tf.initialize_all_variables()
# with tf.Session() as session:
#     session.run(init)
#     conv_op=session.run(conv,feed_dict={x:image})


# dim = conv_op[0,:,:,:].ndim
# conv_img = conv_op[0,:,:,:]
# plt.imshow(conv_img[0,:,:],cmap='Greys')
# plt.show()


