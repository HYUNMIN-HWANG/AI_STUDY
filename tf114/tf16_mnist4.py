# dropout 적용, 다층 레이어 적용
# xavier_initializer etc
# batch size

import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

tf.set_random_seed(777)

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_train.shape) # (60000, 28, 28) (60000,)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000, 10)

#2. Modeling

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

# layer1
# w1 = tf.Variable(tf.random_normal([784, 100],stddev= 0.1, name='weight1'))

# 변수 선언하는 다른 방식 & 가중치 초기화
w1 = tf.get_variable('weight1', shape=[784, 256],
                        initializer=tf.contrib.layers.xavier_initializer()) # kernel initializer >> xavier_initializer(), he_normal   
print("w1 : ", w1)
# w1 :  <tf.Variable 'weight1:0' shape=(784, 100) dtype=float32_ref>

b1 = tf.Variable(tf.random_normal([256]),name='bias1')
print("b1 : ", b1)
# b1 :  <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>

# layer1 = tf.nn.softmax((tf.matmul(x, w)+b))   # activaion=softmax
# layer1 = tf.nn.relu((tf.matmul(x, w)+b))  # activation=relu
# layer1 = tf.nn.selu((tf.matmul(x, w)+b))  # activation=selu   
layer1 = tf.nn.relu(tf.matmul(x, w1)+b1) # activation=elu
print("layer1 : ", layer1)
# layer1 :  Tensor("Elu:0", shape=(?, 100), dtype=float32)

layer1 = tf.nn.dropout(layer1, keep_prob=0.3)   # keep_prob=0.3 : 30% drop out
print("layer1 : ", layer1)
# layer1 :  Tensor("dropout/mul_1:0", shape=(?, 100), dtype=float32)

# layer 2
# w2 = tf.Variable(tf.random_normal([100,50],stddev= 0.1, name='weight2'))
w2 = tf.get_variable('weight2', shape=[256,256],
                        initializer=tf.contrib.layers.xavier_initializer()) # kernel initializer >> xavier_initializer()  
b2 = tf.Variable(tf.random_normal([256]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)   # keep_prob=0.3 : 30% drop out

# layer 3
w3 = tf.get_variable('weight3', shape=[256,128],
                        initializer=tf.contrib.layers.xavier_initializer()) # kernel initializer >> xavier_initializer()  
b3 = tf.Variable(tf.random_normal([128]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.3)   # keep_prob=0.3 : 30% drop out

# layer 4 : output
# w4 = tf.Variable(tf.random_normal([64,10],stddev= 0.1, name='weight4'))
w4 = tf.get_variable('weight4', shape=[128,10],
                        initializer=tf.contrib.layers.xavier_initializer()) # kernel initializer >> xavier_initializer()  
b4 = tf.Variable(tf.random_normal([10],stddev= 0.1, name= 'bias4'))
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)    # activation=softmax

#3. Compile, Train
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)  # 60000 / 100 = 600

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs) :   # 15 epoch
    avg_cost = 0

    for i in range(total_batch) :   # 1 epoch 에 600 번 돈다.
        start = i * batch_size      # 0   100  200  .... 
        end = start + batch_size    # 100 200  300  ....

        batch_x , batch_y = x_train[start:end], y_train[start:end]  # start:end 100개씩 훈련시킨다.
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch   # 전체적인 평균 cost

    print("Epoch :", '%04d' % (epoch + 1), 
          'cost = {:.9f}'.format(avg_cost))

print("== Done ==")

prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

print("ACC : ", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

