import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data")

x_train, y_train = mnist.train.images, mnist.train.labels
xTrainArry=np.array(x_train)

shape=xTrainArry.shape 
#  shape  : (55000, 28, 28, 1)
#  Images : 55000
#  Width  : 28
#  Height : 28
#  Channel: 1

dim = xTrainArry.ndim
arr1 = xTrainArry[0]
arr1Dim = arr1.ndim



print(arr1)