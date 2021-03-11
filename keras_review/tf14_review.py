# tensorflow Deep Learning

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4, 2))
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) 

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

# layer1
w1 = tf.Variable(tf.random_normal([2, 32]), name='weight1')
b1 = tf.Variable(tf.random_normal([32]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# layer2
w2 = tf.Variable(tf.random_normal([32, 8]), name='weight2')
b2 = tf.Variable(tf.random_normal([8]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# layer3
w3 = tf.Variable(tf.random_normal([8, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)

