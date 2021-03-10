import tensorflow as tf

tf.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# y_predict
hypothesis = x_train * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # mse

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        _, w_val, b_val, cost_val = sess.run([train, w, b, cost], \
            feed_dict={x_train:[1,2,3],y_train:[3,5,7]})
        if step % 20 == 0 :
            print(step, cost_val,  w_val, b_val)

    print(sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))    # [12.996066 14.99501  16.993954]
    