from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense 

import numpy as np
from numpy import array

#1. DATA
x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1, activaion='linear'))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4. Evaluate, predict
result = model.evalute(x_test, y_test, batch_size=1)
print(result)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

