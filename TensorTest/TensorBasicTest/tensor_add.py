import tensorflow as tf
import numpy as np

val_1 = np.array([2,3])
val_2 = np.array([4,5])

a = tf.add(val_1,val_2)
b = tf.placeholder(dtype=tf.int32)

hypo = a + b


c = hypo * 3 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run([hypo,c],feed_dict={b:10}))