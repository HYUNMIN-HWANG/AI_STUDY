# fashion_mnist (dnn 구성)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 9
# print(x_train[0].shape)               # (28, 28)

# plt.imshow(x_train[0], 'gray')        # 0 : black, ~255 : white (가로 세로 색깔)
# # plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
# plt.show()  

# x > preprocessing
# print(np.min(x_train),np.max(x_train))  # 0 ~ 255
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255.

print(x_train.shape)    # (60000, 784)
print(x_test.shape)     # (10000, 784)
print(np.min(x_train),np.max(x_train))  # 0.0 ~ 1.0

# y > preprocessing
# print(y_train[:20]) # 0 ~ 9
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(256,input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=70, batch_size=32, validation_split=0.2,verbose=1,callbacks=[es])

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ", np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ", np.argmax(y_pred,axis=1))

# CNN
# loss :  0.3053586483001709
# acc :  0.8960000276565552
# y_test :  [9 1 8 1]
# y_pred :  [9 1 8 1]

# DNN
# loss :  0.3806992769241333
# acc :  0.8955000042915344
# y_test :  [9 1 8 1]
# y_pred :  [9 1 8 1]