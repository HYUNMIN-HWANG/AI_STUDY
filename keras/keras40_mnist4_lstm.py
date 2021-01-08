# 주말과제
# LSTM 모델로 구성 input_shape=(28*28,1) 
# LSTM 모델로 구성 input_shape=(28*14,2)
# LSTM 모델로 구성 input_shape=(28*7,4)
# LSTM 모델로 구성 input_shape=(7*7,16)

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
x_train = x_train.reshape(x_train.shape[0],28*14,2) / 255.  
x_test = x_test.reshape(x_test.shape[0],28*14,2) / 255.    

# print(x_train.shape)    # (60000, 784, 1)
# print(x_test.shape)     # (10000, 784, 1)
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
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(5, input_shape=(28*14,2), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Dropout(0.1))
model.add(Dense(5))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=8, mode='max')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.2, callbacks=[es])

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ")
print(np.argmax(y_test[:10], axis=1))

y_pred = model.predict(x_test[:10])
print("y_predict : ")
print(np.argmax(y_pred,axis=1))

# LSTM
# loss :  1.8663101196289062
# acc :  0.27720001339912415