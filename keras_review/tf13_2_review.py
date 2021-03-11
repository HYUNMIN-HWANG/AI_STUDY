import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

tf.set_random_seed(66)

dataset = load_wine()
x_data = dataset.data
y_data = dataset.target
print(x_data.shape, y_data.shape)   # (178, 13) (178, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, random_state=42, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape)   # (160, 13) (160, 3)

x = tf.placeholder('float', [None,13])
y = tf.placeholder('float', [None,3])

w = tf.Variable(tf.random_normal([13,3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))    # categorical_crossentropy

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(1001) :
        _, cost_val = sess.run([optimizer, cost], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0: 
            print(step, "/ cost : ", cost_val)
    
    y_pred = sess.run(tf.argmax(hypothesis, axis=1), feed_dict={x:x_test})
    real_data = sess.run(tf.argmax(y_test,1))
    print("acc score : ", accuracy_score(real_data, y_pred))

# 1000 / cost :  0.057751615
# acc score :  1.0