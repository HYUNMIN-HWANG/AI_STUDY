from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

#1. DATA
x = np.array(range(1,101)) # 1 ~ 100
y = np.array(range(101,201)) # 101 ~ 200

x_train = x[:60] #0번째부터 59번째까지 ::: 1 ~ 60 
x_val = x[60:80] #60번째부터 79번째까지 ::: 60 ~ 80
x_test = x[80:]  #80번째부터 100번째까지 ::: 81 ~ 100

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1)) #activation 생략할 수 있음
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)


#4. Evalutae, Predict
result = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", result)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)


# mse, mae :  [1.5068799541495537e-07, 0.0003860473516397178]
# RMSE :  0.0003941082622598725
# r2 :  0.999999995328682