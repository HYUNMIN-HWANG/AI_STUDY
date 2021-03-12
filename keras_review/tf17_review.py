# CNN MNIST

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical

tf.set_random_seed(66)

#1. DATA
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

#2. Modeling

# layer 1
w1 = tf.get_variable("w1", shape=[3,3,3,128])
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1)   # Tensor("MaxPool2d:0", shape=(?, 14, 14, 128), dtype=float32)

# layer 2
w2 = tf.get_variable("w2", shape=[3,3,128,64])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)   # Tensor("MaxPool2d_1:0", shape=(?, 7, 7, 64), dtype=float32)

# layer 3
w3 = tf.get_variable("w3", shape=[3,3,64,32])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3 = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)   # Tensor("MaxPool2d_2:0", shape=(?, 4, 4, 32), dtype=float32)

# layer 4
w4 = tf.get_variable("w4", shape=[3,3,32,32])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.elu(L4)
print(L4)   # Tensor("Elu_2:0", shape=(?, 4, 4, 32), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1, L4.shape[1] * L4.shape[2] * L4.shape[3]])
print(L_flat)   # Tensor("Reshape:0", shape=(?, 512), dtype=float32)

# layer 5
w5 = tf.get_variable("w5", shape=[L_flat.shape[1],64], initializer=tf.compat.v1.initializers.he_normal())
b5 = tf.Variable(tf.random_normal([64]))
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)   # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# layer 6
w6 = tf.get_variable("w6", shape=[64, 10], initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L5, w6) + b6)
print(hypothesis)   # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#3. Compile, Train
loss = tf.reduce_mean(-tf.reduce_mean(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.AdamOptimizer(1e-5).minimize(loss)

training_epoch = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)

# Train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epoch) :
    avg_loss = 0

    for i in range(total_batch) :
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        l , _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_loss += l/total_batch
    print("epoch :", '%03d' % (epoch + 1), 'loss = {:.9f}'.format(avg_loss))

# Evaluate, Predict
prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("acc : ", sess.run(acc, feed_dict={x:x_test, y:y_test}))
