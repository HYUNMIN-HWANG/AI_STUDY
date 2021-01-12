# 저장한 numpy 불러오기 : np.load

import numpy as np
m_x_train = np.load('../data/npy/mnist_x_train.npy')
m_x_test = np.load('../data/npy/mnist_x_test.npy')
m_y_train = np.load('../data/npy/mnist_y_train.npy')
m_y_test = np.load('../data/npy/mnist_y_test.npy')

# print(x_data)
# print(y_data)
print(m_x_train.shape)  # (60000, 28, 28)
print(m_x_test.shape)   # (10000, 28, 28)
print(m_y_train.shape)  # (60000,)
print(m_y_test.shape)   # (10000,)

# =========================== 모델을 완성하시오 ===========================

# preprocessing
x_train = m_x_train.reshape(60000, 28, 28, 1)/255. 
x_test = m_x_test.reshape(10000, 28, 28, 1)/255. 

# y >> OnHotEncoding

from sklearn.preprocessing import OneHotEncoder

y_train = m_y_train.reshape(-1,1)
y_test = m_y_test.reshape(-1,1)
# print(m_y_train[0])       # [5]
# print(m_y_train.shape)    # (60000, 1)
# print(m_y_test[0])        # [7]
# print(m_y_test.shape)     # (10000, 1)

encoder = OneHotEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()    #toarray() : list 를 array로 바꿔준다.
# print(y_train)
# print(y_test)
# print(y_train.shape)    # (60000, 10)
# print(y_test.shape)     # (10000, 10)


#2. Modling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=16, kernel_size=(4,4), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.1))
model.add(Flatten())

model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

# model.summary()

# Compile, Train

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 체크포인트의 가중치를 저장할 파일경로 지정
modelpath='../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
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

# Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("accuracy : ", result[1])


# 응용
# y_test 10개와 y_test 10개를 출력하시오

# print("y_test[:10] :\n", y_test[:10])
print("y_test[:10] :")
print(np.argmax(y_test[:10],axis=1))

y_predict = model.predict(x_test[:10])
print("y_pred[:10] :")  
print(np.argmax(y_predict,axis=1))

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

# loss :  0.034563612192869186
# acc :  0.9889000058174133
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 5 9]