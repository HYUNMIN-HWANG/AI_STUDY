# CNN
# mnist

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)
print(y_train.shape)    # (60000,)
print(y_test.shape)     # (10000,)

# print(x_train[0])
# print(y_train[0])       # 5

# plt.imshow(x_train[0],'gray')
# plt.show()

# x > preprocessing
# print(np.min(x_train[0]))   # 최솟값 0 
# print(np.max(x_train[0]))   # 최댓값 255
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.
print(x_train.shape)    # (60000, 28, 28, 1)
print(x_test.shape)     # 10000, 28, 28, 1)

# y > preprocessing
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()

print(y_train.shape)    # (60000, 10)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=2,padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=2, validation_split=0.2)

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ", np.argmax(y_test[:10],axis=1))
y_pred = model.predict(x_test[:10])
print("y_pred : ", np.argmax(y_pred[:10], axis=1))
