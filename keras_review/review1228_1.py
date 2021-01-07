from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import Dense 

import numpy as np
from numpy import array

#1. DATA
x_train = np.array([1,2,3,4])
y_train = np.array([1,2,3,4])

x_test = np.array([5,6,7,8,9])
y_test = np.array([5,6,7,8,9])

x_pred = np.array([10,11,12])

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", result)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

