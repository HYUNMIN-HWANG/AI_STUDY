# 아래 모델을 완성하시오 (지표는 acc >= 0.985)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

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
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
# 4차원 만들어준다. float타입으로 바꾸겠다. -> /255. -> 0 ~ 1 사이로 수렴됨
x_test = x_test.reshape(10000, 28, 28, 1)/255. 
# x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],1)

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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=(4,4), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.1))
model.add(Flatten())

model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

# model.summary()

# Compile, Train
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='acc', patience=5, mode='max')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
model.fit(x_train, y_train, epochs=9, batch_size=64, validation_split=0.3)

# Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
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
 
# loss :  0.034563612192869186
# acc :  0.9889000058174133
# y_test[:10] :
# [7 2 1 0 4 1 4 9 5 9]
# y_pred[:10] :
# [7 2 1 0 4 1 4 9 5 9]