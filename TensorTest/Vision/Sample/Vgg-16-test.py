import tensorflow as tf


#fixed size image 224 * 224
# conv filter size 3x3
# 각 픽셀의 RGB값을 Substracting 
# padding 1px
# Spatial Pooling is carried out by five max-pooling layers, which follow some of the conv.layers
# max pooling perfomred over 2*2 window , stride 2

# A stack of convolutional layers (which has a different depth in different architectures) is followed by
# three Fully-Connected (FC) layers: the first two have 4096 channels each, the third performs 1000-
# way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is
# the soft-max layer.

def Layer():
    x = tf.placeholder(tf.float32,[1,224,224,3])
    layer1_w = tf.Variable(tf.truncated_normal([3,3,3,64]))
    layer1 = tf.nn.conv2d(x,layer1_w,strides=[1,1,1,1],padding='VALID')


