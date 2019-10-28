from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 10
BATCH_SIZE = 128

def load_data():
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    x_test, y_test = mnist.test.images, mnist.test.labels

    assert(len(x_train) == len(y_train))
    assert(len(x_validation) == len(y_validation))
    assert(len(x_test) == len(y_test))

    print()
    print("Image Shape: {}".format(x_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(x_train)))
    print("Validation Set: {} samples".format(len(x_validation)))
    print("Test Set:       {} samples".format(len(x_test)))

    return x_train,x_validation,x_test,y_train,y_validation,y_test

def pad(x_train,x_test,x_validation):
    x_train = np.pad(x_train,((0,0),(2,2),(2,2),(0,0)),'constant')
    y_validation = np.pad(x_validation,((0,0),(2,2),(2,2),(0,0)),'constant')
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)),'constant')

    print("updated image shape:{}".format(x_train[0].shape))

def visualize_data(x_train,y_train):
    index = random.randint(0, len(x_train))
    image = x_train[index].squeeze()

    plt.figure(figsize=(1,1))
    plt.imshow(image, cmap="gray")
    #plt.show()
    print(y_train[index])

def preprocess_data(x_train,y_train):
    x_train, y_train = shuffle(x_train, y_train)

def LeNet(x):
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }
    

    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6], mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev= sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding='VALID')

    fc1 = flatten(pool_2)

    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w)+fc1_b

    fc1 = tf.nn.relu(fc1)

    fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), mean=mu, stddev=sigma))
    fc2_b=tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    return logits

def feature_labels():
    x=tf.placeholder(tf.float32,(None,32,32,1))
    y=tf.placeholder(tf.int32,(None))
    one_hot_y = tf.one_hot(y,10)

    return x,y,one_hot_y

def training_pipline(x,one_hot_y):
    rate = 0.001
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    return logits,one_hot_y,training_operation

def model_evaluation(logits,one_hot_y):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y,1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
 
    return saver,accuracy_operation

def evaluate(x_data, y_data,accuracy_operation):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy + len(batch_x))

    return total_accuracy / num_examples

def train_model(x_train,y_train,training_operation,saver,x_validation,y_validation,accuracy_operation,x,y):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples=len(x_train)

        print("Training..")
        print()
        for i in range(EPOCHS):
            x_train,y_train = shuffle(x_train, y_train)
            for offset in range(0, num_examples,BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
            validation_accuracy = evaluate(x_validation, y_validation,accuracy_operation)

        print("EPOCH {}...".format(i+1))
        print("Validation Accuracy={:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, 'lenet')
    print("Model saved")

def evaludate_model(saver):
    with tf.Session() as sess:
        saver.restore

        test_accuracy = evaluate(x_Test, y_test)


    
def main():
    x_train,x_validation,x_test,y_train,y_validation,y_test=load_data()
    pad(x_train,x_test,x_validation)
    visualize_data(x_train,y_train)
    x,y,one_hot_y=feature_labels()

    logits,one_hot,training_operation=training_pipline(x=x,one_hot_y=one_hot_y)

    saver,accuracy_operation = model_evaluation(logits,one_hot)

    train_model(x_train,y_train,training_operation,saver,x_validation,y_validation,accuracy_operation,x,y)



if __name__ == "__main__":
    main()

