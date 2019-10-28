import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten

EPOCHS = 50
BATCH_SIZE = 128



def resize_imgs(imgs):
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
        #... => ellipsis == [i,:,:,0] 과 비슷한 의미다.

    return resized_imgs

def resize_img(img):
    img = np.reshape(img,[1,28,28,1])
    resizedImg = np.zeros([1,32,32,1])
    resizedImg[0,:,:,0] = transform.resize(img[0,:,:,0],(32,32))

    return resizedImg

def Lenet_test(img):
    resizeImg = resize_img(img).astype(np.float32)
    x = tf.placeholder(tf.float32,[1,32,32,1])
    w1 = tf.Variable(tf.random_normal([5,5,1,6],mean=0,stddev=0.1))
    conv1_bias = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(resizeImg,w1,strides=[1,1,1,1], padding='VALID') + conv1_bias
    pool_1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    conv2_w = tf.Variable(tf.truncated_normal([5,5,6,16],mean=0,stddev=0.1))
    conv2_bias = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1,conv2_w,strides=[1,1,1,1],padding='VALID') + conv2_bias

    pool_2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    init = tf.initialize_all_variables()
    # with tf.Session() as session:
    #     session.run(init)
    #     conv_op = session.run(pool_2,feed_dict={x:resizeImg})
    #     session.close()

    # for i in range(1,conv_op.shape[3]):
    #     rslt = conv_op[0,:,:,i]
    #     plt.imshow(rslt,cmap="Greys")
    #     plt.show()

def Lenet(img):

    conv1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6],mean=0,stddev=0.01))
    conv1_bias = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(img,conv1_w,strides=[1,1,1,1], padding='VALID') + conv1_bias
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

    return logits

def pad_test(img):
    rslt = np.reshape(img,[28,28])
    npad = ((2,2),(2,2))
    data_padding = np.pad(rslt,npad,'constant',constant_values=(0))

    plt.imshow(data_padding,cmap='Greys')
    plt.show()

def pad(x):
    rslt = np.pad(x,((0,0),(2,2),(2,2),(0,0)),'constant')
    return rslt

def feature_labels():
    x=tf.placeholder(tf.float32,(None,32,32,1))
    y=tf.placeholder(tf.int32,(None))
    one_hot_y = tf.one_hot(y,10)

    return x,y,one_hot_y


def train(x_train,y_train,training_operation,x,y):
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
    return sess

def training_pipline(x,one_hot_y):
    rate = 0.001
    logits = Lenet(x)
    crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_y)
    loss = tf.reduce_mean(crossEntropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss)

    return logits,one_hot_y,training_operation

def visualize_data(x_train,y_train):
    index = random.randint(0,len(x_train))
    image = x_train[index].squeeze()
    plt.figure(figsize=(1,1))
    plt.imshow(image, cmap="gray")
    print(y_train[index])
    plt.show()

def main():
    mnist = input_data.read_data_sets("MNIST_data",reshape=False)

    x_train, y_train = mnist.train.images, mnist.train.labels
    x_train = pad(x_train)
    x,y,one_hot_y = feature_labels() #place holder 생성

    logits,one_hot_y,training_operation = training_pipline(x,one_hot_y)
    sess=train(x_train,y_train,training_operation,x,y)

    for r in range(11):
        testDigit = mnist.train.images[r]
        testDigit = np.reshape(testDigit,[1,28,28,1]).astype(np.float32)
        testDigit = np.pad(testDigit,((0,0),(2,2),(2,2),(0,0)),'constant')
        rslt = sess.run(tf.argmax(logits,1),feed_dict={x:testDigit})
        print("Prediction:",rslt)
        plt.imshow(testDigit[0,:,:,0],cmap='Greys')
        plt.show()
      
    #digit = mnist.train.images[7]
    #pad_test(digit)
    #Lenet_test(digit)

if __name__ == "__main__":
    main()
# tensorflow의 변수형은 그래프를 실행하기 전에 초기화를 해줘야 그 값이 변수에 지정된다.
# initializer를 선언 하기 전에는 y값이 지정되어 있지 않다는 의미.
# tensorflow 변수는 global 하게 유지 되어야한다?