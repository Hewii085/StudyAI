import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

arry =np.zeros([64,64],dtype=np.float32)

# startIdx = 13
# for x in range(0,25):
#     arry[x][startIdx] = 1

arry[13][13] = 1
arry[14][14] = 1
arry[15][15] = 1
arry[16][16] = 1
arry[17][17] = 1
arry[19][18] = 1
arry[20][18] = 1
arry[21][18] = 1
arry[22][18] = 1
arry[23][18] = 1
arry[24][18] = 1

# plt.imshow(arry,cmap="Greys")
# plt.show()

arry = np.reshape(arry,[1,64,64,1])

X = tf.placeholder(tf.float32,[1,64,64,1])
W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
L1 = tf.nn.conv2d(arry,W1,strides=[1,1,1,1],padding="SAME")
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
L1 = tf.layers.batch_normalization(L1,training=True,momentum=0.99,epsilon=0.001,center=True,scale=True)
L1 = tf.nn.leaky_relu(L1,alpha=0.1)

# W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
# L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding="SAME")
# L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
# L2 = tf.layers.batch_normalization(L2,training=True,momentum=0.99,epsilon=0.001,center=True,scale=True)
# L2 = tf.nn.leaky_relu(L2,alpha=0.1)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    conv_op=session.run(L1,feed_dict={X:arry})

check = conv_op[0,0,0,0]
for i in range(1,conv_op.shape[3]):
    aa = conv_op[0,:,:,i] #
    plt.imshow(aa,cmap="Greys")
    plt.show()