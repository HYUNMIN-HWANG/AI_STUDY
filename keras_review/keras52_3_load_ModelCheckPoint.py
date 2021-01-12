# model save 와 weight save 비교하기

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)  # (60000,)        (10000,)

# x > preprocessing
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1) / 255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1) / 255.

# y > preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Modeling
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = load_model('../data/modelcheckpoint/k51_test_10_0.0177.h5')

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

