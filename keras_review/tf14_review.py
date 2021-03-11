# tensorflow Deep Learning

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4, 2))
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) 

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

# layer1 : input
w1 = tf.Variable(tf.random_normal([2, 32]), name='weight1')
b1 = tf.Variable(tf.random_normal([32]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# layer2
w2 = tf.Variable(tf.random_normal([32, 8]), name='weight2')
b2 = tf.Variable(tf.random_normal([8]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# layer3 : output
w3 = tf.Variable(tf.random_normal([8, 1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 

train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) 

    for step in range (20001) :
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 :
            print(step, "/ cost : ", cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})
    print("hypothesis : ", h, "/ predict : ", c, "/ acc : ", a)
