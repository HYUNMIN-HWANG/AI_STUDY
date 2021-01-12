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

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',\
    input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))
# model.summary()

#3. Compile, Train
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

# 1.
model.load_weights('../data/h5/k52_test_weight.h5')

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

#2.
model2 = load_model('../data/h5/k51_test_model_2.h5')
result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result2[0])
print("acc : ", result2[1])