# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)  > 0 ~ 9 다중 분류

# print(x_train[0])   
# print("y_train[0] : " , y_train[0])   # 5
# print(x_train[0].shape)               # (28, 28)

# plt.imshow(x_train[0], 'gray')  # 0 : black, ~255 : white (가로 세로 색깔)
# plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
# plt.show()  

# preprocessing
x_train = x_train/255. 
x_test = x_test/255. 

# y >> OnHotEncoding

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
# print(y_train[0])       # [5]
# print(y_train.shape)    # (60000, 1)
# print(y_test[0])        # [7]
# print(y_test.shape)     # (10000, 1)

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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, padding='same', strides=1, input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Conv1D(filters=16, kernel_size=2))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32, kernel_size=2))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))


model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

# model.summary()

# Compile, Train

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 체크포인트의 가중치를 저장할 파일경로 지정
modelpath = '../data/modelcheckpoint/k54_7_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')

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

# Conv2D
# loss :  0.034563612192869186
# acc :  0.9889000058174133
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 5 9]

# Conv1D
# loss :  0.24231469631195068
# accuracy :  0.9302999973297119
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 4 6 9]