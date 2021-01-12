# CNN
# fashion_mnist
# from tensorflow.keras.callbacks import ModelCheckpoint

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
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.

print(x_train.shape)    # (60000, 28, 28, 1)
print(x_test.shape)     # (10000, 28, 28, 1)
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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=112, kernel_size=(2,2),padding='same',strides=1,input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=84,kernel_size=(2,2)))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=28,kernel_size=(2,2)))
model.add(Conv2D(filters=28,kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10,activation='softmax'))

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 체크포인트의 가중치를 저장할 파일경로 지정
modelpath='../data/modelcheckpoint/k46_1_fashion_{epoch:02d}-{val_loss:.4f}.hdf5'
                                        # 02d : 정수 두 자리만 적겠다. / .4f : 소수점 아래 4째자리까지 적겠다.
                                        # 저장 예시) k45_mnist_37-0.0100.hdf5
                                        # 저장된 파일 중에 가장 마지막에 생성된게 가장 좋은 것이 됨
es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')
                    # filepath : 최저점이 찍힐 때마다 가중치가 세이브된 파일이 생성된다. 
                    # 궁극의 목적 : 최적의 weight를 구하기 위해서
                    # predict할 때 혹은 evaluate 할 때 이 weight를 넣기만 하면된다.
                    
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es, cp])

#4. Evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print("y_test : ", np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ", np.argmax(y_pred,axis=1))


# 시각화
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6))  # 판 사이즈 (가로 10, 세로 6)

# plt.subplot(2, 1, 1)    # plot : 도화지 하나에 그림을 그리겠다.
#                         # 2행 1열 중 첫 번째
#                         # 만약 (3, 1, 1) 이라면 세 개의 plot이 있어야 한다. (3, 1, 1) (3, 1, 2) (3, 1, 3)
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()

# # plt.title('손실비용') # 과제 : 한글 깨짐 오류 해결할 것
# plt.title('Cost Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째
# plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
# plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
# plt.grid()              # 모눈종이 격자위에 그리겠다.

# # plt.title('정확도')   # 과제 : 한글 깨짐 오류 해결할 것
# plt.title('Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.show()

# CNN
# loss :  0.3053586483001709
# acc :  0.8960000276565552
# y_test :  [9 1 8 1]
# y_pred :  [9 1 8 1]

# ModelCheckPoint
# loss :  0.32927361130714417
# acc :  0.8809999823570251
# y_test :  [9 1 8 1]
# y_pred :  [9 1 8 1]