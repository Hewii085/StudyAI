import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import random
from PIL import Image


def show_image(img,conv_op):

    with tf.Session() as sess :
        gs1 = gridspec.GridSpec(1,3)
        plt.subplot(gs1[0,0])
        plt.axis('off')
        plt.imshow(img[:,:,:])

        plt.subplot(gs1[0,1])
        plt.axis('off')
        plt.imshow(conv_op[0,:,:,:])
    # plt.imshow(max_pool_op[0,:,:,:])
        plt.show()

dirPath = "H:\\test"
fileNames = os.listdir(dirPath)
flt = tf.Variable(tf.random_normal([5, 5, 3, 1], stddev=0.01),dtype="float32")

for fileName in fileNames:
    img = Image.open(os.path.join(dirPath,fileName))

    x = tf.placeholder(tf.float32, [None,img.width*img.height*3])
    x_img = tf.reshape(x,[-1,img.width,img.height,3])
    conv = tf.nn.conv2d(x_img,flt,strides=[1,1,1,1], padding='SAME')
    
    show_image(img,conv)


    
