# tensorflow Deep Learning

import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)  # (4, 2))
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) 

x = tf.placeholder(tf.float32, shape=[None,2])
