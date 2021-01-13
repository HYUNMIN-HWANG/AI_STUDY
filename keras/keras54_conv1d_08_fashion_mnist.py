# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교

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
x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape)    # (60000, 28, 28)
print(x_test.shape)     # (10000, 28, 28)
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
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3,padding='same',strides=1,\
    activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

# model.summary()

#3. Compile, Train

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = '../data/modelcheckpoint/k54_8_fa_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')

model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es, cp])

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ", np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ", np.argmax(y_pred,axis=1))

# CNN
# loss :  0.23155026137828827
# acc :  0.9233999848365784
# y_test :  [9 1 8 1]
# y_pred :  [9 1 8 1]

# Conv1D
# loss :  0.3514102101325989
# acc :  0.8676000237464905
# y_test :  [9 1 8 1]
# y_pred :  [9 1 8 1]