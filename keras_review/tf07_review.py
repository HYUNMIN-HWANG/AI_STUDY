import tensorflow as tf

tf.compat.v1.set_random_seed(777)

# [1]
W = tf.Variable(tf.random_normal([1]), name='weight')

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

aaa = sess.run(W)
print("aaa :",aaa)
sess.close()

# [2]

W = tf.Variable(tf.random_normal([1]), name='weight')

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

bbb = W.eval()
print("bbb :", bbb)

# [3]

W = tf.Variable(tf.random_normal([1]), name='weight')

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

ccc = W.eval(session=sess)
print("ccc :",ccc)