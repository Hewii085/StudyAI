import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image
import numpy

img = Image.open("H:\\20180723010332379.png")
plt.imshow(img)

def test_func(img,kernel):
    with tf.Graph().as_default():
        num_maps=3

        img=numpy.asarray(img,dtype='float32')/256.
        img_shape = img.shape
        img_reshaped = img.reshape(1,img_shape[0],img_shape[1],num_maps)

        x = tf.placeholder('float32',[1,None,None,num_maps])
        w = tf.get_variable('w',initializer=tf.to_float(kernel))

        #[1,3,3,1] 1*3*3 conv, 32 output
        weight = tf.Variable(tf.random_normal([3,3,3,3],stddev=0.01))

        conv = tf.nn.conv2d(x,w,strides=[1,3,3,1],padding='SAME')
        sig = tf.sigmoid(conv)
        max_pool = tf.nn.max_pool(conv,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME')

        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            conv_op,sigmoid_op, max_pool_op = session.run([conv,sig,max_pool],
                                                       feed_dict={x: img_reshaped})

        gs = gridspec.GridSpec(1,2)
        plt.subplot(gs[0,0]); plt.axis('off'); plt.imshow(img[:,:,:])
        plt.subplot(gs[0, 1]); plt.axis('off'); plt.imshow(conv_op[0, :, :, :])

        plt.show()

def test_func2():
    with tf.Graph().as_default:
        


a = np.zeros([3,3,3,3])
a[1, 1, :, :] = 5
a[0, 1, :, :] = -1
a[1, 0, :, :] = -1
a[2, 1, :, :] = -1
a[1, 2, :, :] = -1

test_func(img,a)