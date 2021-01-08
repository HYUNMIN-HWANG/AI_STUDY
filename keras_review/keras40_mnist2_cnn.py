import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)
# print(y_train.shape)    # (60000,)
# print(y_test.shape)     # (10000,)

# print(x_train[0])
# print(y_train[0])   #--> 5

# x > preprocessing
# print(np.min(x_train))  # --> 0
# print(np.max(x_train))  # --> 255
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1) / 255.  # (60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1) / 255.       # (10000, 28, 28, 1)

# print(np.min(x_train))  # --> 0.0
# print(np.max(x_train))  # --> 1.0

# y > preprocessing
# print(y_train[:10])       # --> 0 부터 9까지 다중분류

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train[:10])
# print(y_test[:10])
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=2, padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.1))
model.add(Conv2D(10, (2,2), padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ")
print(np.argmax(y_test[:10],axis=1))

y_pred = model.predict(x_test[:10])
print("x_test : ")
print(np.argmax(y_pred, axis=1))