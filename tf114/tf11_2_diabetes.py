# 회귀
# 최종 sklearn의 r2_score 값으로 결론 낼 것! 

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score , accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

tf.set_random_seed(66)

dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape) # (442, 10) (442, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=42)
print(x_train.shape, x_test.shape)  # (397, 10) (45, 10)

scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) 

    for step in range(5001) :
        _, cost_val, hyp_val = sess.run([train, cost, hypothesis], feed_dict={x:x_train, y:y_train})
        if step % 20 == 0 :
            print(step, cost_val)
    y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    print("r2 : ", r2_score(y_test, y_predict))

# r2 :  0.3091480858805067