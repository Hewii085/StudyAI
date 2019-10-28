import tensorflow as tf
import cv2


saver = tf.train.Saver()

with tf.Session as sess:
    saver.restore(sess,".\\save\\model.data-00000-of-00001")






