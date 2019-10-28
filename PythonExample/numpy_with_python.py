import tensorflow as tf
import numpy as np

"""
numpy와 tensorflow와의 연산을 테스트
"""

npArry = [[1,1],[1,1]]
inputData = [[2,2],[2,2]]
x = tf.placeholder(tf.int32,[2,2])
y = x+npArry #Bias 형태인가?

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    rslt = sess.run(y,feed_dict={x:inputData})
    print(rslt)

print(y)
