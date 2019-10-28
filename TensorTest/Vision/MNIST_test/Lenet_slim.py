import tensorflow as tf
from tensorflow.models.research.slim.nets import lenet
from tensorflow.examples.tutorials.mnist import input_data

# def lenet(images, num_classes=10, is_training=False,
#           dropout_keep_prob=0.5,
#           prediction_fn=slim.softmax,
#           scope='LeNet'):
X = tf.placeholder(tf.float32, [None, 28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

mnist = input_data.read_data_sets("MNIST_data",one_hot=True,reshape=False)
x_train, y_train = mnist.train.images, mnist.train.labels
# x_train data rank 2로 reshape 필요
logits,endPoints= lenet.lenet(images=x_train,num_classes=10,is_training=False,dropout_keep_prob=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

#버그 수정해야함 Chunk 어쩌고 저쩌고
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    for i in range(20):
        total_batch = int(mnist.train.num_examples / 100)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(100)
            feed_dict = {X:batch_x, Y:batch_y}
            c = sess.run([cost,optimizer],feed_dict=feed_dict)
        print('Epoch')

print('Learning Finished!')





