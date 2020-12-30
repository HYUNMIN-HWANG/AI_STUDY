# metrics = ['accuracy'] ['mae'] ['mse']

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. Data 
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. Model
model = Sequential()

model.add(Dense(50000, input_dim=1, activation='relu'))
model.add(Dense(1000))     
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) # output = 1 

#3. Compile, Train
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # 출력결과 : accuracy == 0.0
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])      # 출력결과 : mse == loss
model.compile(loss='mse', optimizer='adam', metrics=['mae'])        # mae : 평균 절대 오차

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)  #loss 'mse' 와 metrics 'mae' 출력됨

# result = model.predict([9])
result = model.predict(x_train)

print("result : ", result)