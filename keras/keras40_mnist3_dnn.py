# 주말과제
# Dense 모델로 구성 input_shape=(28*28,)
# CNN을 이겨라

# 아래 모델을 완성하시오 (지표는 acc >= 0.985)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,) 

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 5
# print(x_train[0].shape)               # (28, 28)

# x > preprocessing
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255. 
# 4차원 만들어준다. float타입으로 바꾸겠다. -> /255. -> 0 ~ 1 사이로 수렴됨
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255. 

# print(x_train.shape)    # (60000, 784)
# print(x_test.shape)     # (10000, 784)

# y > preprocessing
# print(y_train)
# print(y_test)
# print(y_train.shape)    # (60000, )
# print(y_test.shape)     # (10000, )

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Dense(196, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(56))
model.add(Dropout(0.2))
model.add(Dense(56))
model.add(Dense(28))
model.add(Dense(10, activation='softmax'))
model.summary()

# Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])
# model.fit(x_train, y_train, epochs=9, batch_size=64, validation_split=0.3)

# Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ",loss)
print("acc : ",acc)


# 응용
# y_test 10개와 y_test 10개를 출력하시오

# print("y_test[:10] :\n", y_test[:10])
print("y_test[:10] :")
print(np.argmax(y_test[:10],axis=1))

y_predict = model.predict(x_test[:10])
print("y_pred[:10] :")  
print(np.argmax(y_predict,axis=1))

# CNN
# loss :  0.034563612192869186
# acc :  0.9889000058174133
# y_test[:10] : [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] : [7 2 1 0 4 1 4 9 5 9]

# DNN
# loss :  0.11503560841083527
# acc :  0.9812999963760376
# y_test[:10] : [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] : [7 2 1 0 4 1 4 9 6 9]