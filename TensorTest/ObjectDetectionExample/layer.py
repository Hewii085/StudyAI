import tensorflow as tf
import numpy as np

class TestModel:

    def __init__(self,epoch,batchSize):
        self._epoch = epoch
        self._batchSize = batchSize

    def test_layer(self):
    #image size??
        self.x = tf.placeholder(tf.float32,[None,250,250,3])
        self.yBox = tf.placeholder(tf.float32,[None,4])
        self.yObj = tf.placeholder(tf.float32,[None,1])

        with tf.name_scope("model") as scope:
            w1 = tf.Variable(tf.random_normal([3,3,3,32],stddev=0.01))
            layer1 = tf.nn.conv2d(self.x,w1,strides=[1,1,1,1],padding="SAME")
            layer1 = tf.nn.max_pool(layer1,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            layer1 = tf.layers.batch_normalization(layer1,training=True,momentum=0.99,epsilon=0.001,center=True,scale=True)
            layer1 = tf.nn.leaky_relu(layer1,alpha=0.1)

            w2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
            layer2 = tf.nn.conv2d(layer1,w2,strides=[1,1,1,1],padding="SAME")
            layer2 = tf.nn.max_pool(layer2,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            layer2 = tf.layers.batch_normalization(layer2,training=True,momentum=0.99,epsilon=0.001,center=True,scale=True)
            layer2 = tf.nn.leaky_relu(layer2,alpha=0.1)

            w3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
            layer3 = tf.nn.conv2d(layer2,w3,strides=[1,1,1,1],padding="SAME")
            layer3 = tf.nn.max_pool(layer3,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            layer3 = tf.layers.batch_normalization(layer3,training=True,momentum=0.99,epsilon=0.001,center=True,scale=True)
            layer3 = tf.nn.leaky_relu(layer3,alpha=0.1)

            #reshape 필요 pool 횟수등 연산해서 마지막 output에 맞도록 하여 1-D형태로 reshape 필요

            modelBox = tf.layers.dense(inputs=layer3,units=4)
            modelHasObj = tf.layers.dense(inputs=layer3,units=1,activation=tf.nn.sigmoid)

        with tf.name_scope("loss_func") as scope:
            lossObj = tf.losses.log_loss(labels=self.yObj,predictions = modelHasObj)
            lossBox = tf.losses.huber_loss(labels=self.yBox, predictions = modelBox)

            batchSize = tf.case(tf.shape(self.yObj)[0], tf.float32)
            numObjLabel = tf.cast(tf.count_nonzero(tf.cast(self.yObj > 0.0, tf.float32)),tf.float32)

            ratioHasObjects = (numObjLabel * tf.constant(100.0)) / batchSize
            loss = lossObj + (lossBox*ratioHasObjects)

            tf.summary.scalar("loss",loss)
            tf.summary.scalar("loss_bbox",lossBox)
            tf.summary.scalar("lossObj",lossObj)
    
        with tf.name_scope("optimizer") as scope:
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        op = tf.summary.merge_all()

        return op,self.x,self.yBox,self.yObj

    def train(self,imageSet,metaSet):
        init = tf.global_variables_initializer()
        self.test_layer()
        #saver = tf.train.Saver()

        with tf.Session() as sess:
            #writer = tf.summary.FileWriter("./logs/loc_logs",sess.graph)
            sess.run(init)

            for i in range(2000):
                sess.run(self.optimizer,feed_dict={self.x : imageSet,
                                                   self.yObj :metaSet[:,0],
                                                   self.yBox :metaSet[:,1:]})
                print(i)
                                                
            #batchsize / epoch 필요

