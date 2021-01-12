# model save 와 weight save 비교하기
# 결과 >> 모델을 save한 것과 weight를 save한 것은 결과 동일함.

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 
# print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  > 0 ~ 9 다중 분류

# x >> preprocessing
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. 
x_test = x_test.reshape(10000, 28, 28, 1)/255. 

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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model2 = Sequential()
model2.add(Conv2D(filters=16, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model2.add(MaxPooling2D(pool_size=2))
model2.add(Dropout(0.1))
model2.add(Conv2D(filters=16, kernel_size=(4,4), padding='same', strides=1))
model2.add(MaxPooling2D(pool_size=3))
model2.add(Dropout(0.1))
model2.add(Flatten())
model2.add(Dense(8))
model2.add(Dense(10, activation='softmax'))

# (1) 모델링 하고 난 직후 model.save
# model.save('../data/h5/k52_1_model1.h5')

# k52_1_MCK_0.0589.hdf5 와 비교

#3 Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath='../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5, mode='max')
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='auto')

model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[es]) #, cp])

# (2) 컴파일, 훈련한 후 model.save
# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

# ============ save 비교하기 ============

# model1 : 모델과 가중치를 불러온다.
model1 = load_model('../data/h5/k52_1_model2.h5') 

# 4-1 Evaluate, Predict
result = model1.evaluate(x_test, y_test, batch_size=32)
print("model1_loss : ", result[0])
print("model1_accuracy : ", result[1])
# model1
# model1_loss :  0.055421654134988785
# model1_accuracy :  0.9837999939918518

# model2 : 
model2.load_weights('../data/h5/k52_1_weight.h5')

# 4-2 Evaluate, Predict
result2 = model1.evaluate(x_test, y_test, batch_size=32)
print("model2_loss : ", result2[0])
print("model2_accuracy : ", result2[1])

# model2_loss :  0.055421654134988785
# model2_accuracy :  0.9837999939918518