import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

for step in range(3) :
    sess.run(train)
    print(step, sess.run(cost), sess.run(W), sess.run(b))