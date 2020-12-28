#import는 통상적으로 맨 위에 몰아서 넣어준다.
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
#[2] from tensorflow.keras import models
#[3] from tensorflow import keras

from tensorflow.keras.layers import Dense


#1. Data 
#원래의 데이터를 훈련시키는 데이터와 평가 데이터를 구분한다. 1,2,3,4,5,6,7,8을 둘로 나눈 것
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])

x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113])

#2. Model
model = Sequential()
#[2] model = models.Sequential()
#[3] model = keras.models.Sequential()

model.add(Dense(250, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss) #loss 값을 줄여라

result = model.predict(x_predict)
print("result : ", result)