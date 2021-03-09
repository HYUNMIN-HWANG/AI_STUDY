import tensorflow as tf
print(tf.__version__)   # 1.14.0

hello = tf.constant("Hello Python")
print(hello)

sess = tf.Session()
print(sess.run(hello))

node1 = tf.constant(15.0, tf.float32)
node2 = tf.constant(8.0)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
print(sess.run(adder_node, feed_dict={a:3, b:2}))


x = tf.Variable([5], dtype=tf.float32, name='train')
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(x))  # [5.]
# node_add = tf.add(node1, node2)
# print("ADD : ", sess.run(node_add))

# node_subtract = tf.subtract(node1, node2)
# print("Subtract : ", sess.run(node_subtract))

# node_multiply = tf.multiply(node1, node2)
# print("Multiply : ", sess.run(node_multiply))

# node_divide = tf.divide(node1, node2)
# print("Divide : ", sess.run(node_divide))