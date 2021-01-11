# cifar100 (컬러) - CNN
# from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np

#1. DATA 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 19
# print(x_train[0].shape)               # (32, 32, 3)

# plt.imshow(x_train[0])        # 0 : black, ~255 : white (가로 세로 색깔)
# # plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
# plt.show()  

# x > preprocessing
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]) / 255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]) / 255.
# print(x_train.shape)    # (50000, 32, 32, 3)

# y > preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),padding='same',\
    activation='relu',input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(Dropout(0.3))
model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=2))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='softmax'))

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath='./modelCheckpoint/k46_3_cifar100_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='loss', patience=10, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,validation_split=0.2, verbose=1, callbacks=[es,cp])
# model.fit(x_train, y_train, epochs=100, batch_size=32,validation_split=0.2, verbose=1)

#4. predict, Evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ", np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ", np.argmax(y_pred,axis=1))


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))  # 판 사이즈 (가로 10, 세로 6)

plt.subplot(2, 1, 1)    # plot : 도화지 하나에 그림을 그리겠다.
                        # 2행 1열 중 첫 번째
                        # 만약 (3, 1, 1) 이라면 세 개의 plot이 있어야 한다. (3, 1, 1) (3, 1, 2) (3, 1, 3)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

# plt.title('손실비용') # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째
plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.grid()              # 모눈종이 격자위에 그리겠다.

# plt.title('정확도')   # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# CNN
# loss :  4.324522018432617
# acc :  0.2232999950647354
# y_test :  [83 14 51 42]
# y_pred :  [88 80 51 44]


# ModelCheckPoint
# loss :  4.961197853088379
# acc :  0.2004999965429306
# y_test :  [83 14 51 42]
# y_pred :  [66 74 44 42]