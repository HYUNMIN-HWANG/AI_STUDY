import tensorflow as tf  
import numpy as np
from numpy import array

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA
x_train = array([1,2,3,4,5,6,7,8,9,10])
y_train = array([1,2,3,4,5,6,7,8,9,10])

x_test = array([11,12,13,14,15,16])
y_test = array([11,12,13,14,15,16])

x_pred = array([17,18,19,20])

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile & Train
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. Evalutae & Predict
result = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", result)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))

