# 실습 train과 test data를 분리해서 소스를 완성하시오

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array([range(100), range(301,401), range(1, 101)])
y = np.array(range(711,811))
print(x.shape)  #(3, 100)

x = np.transpose(x)
print(x.shape)  #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=55)
print(x_train.shape)    #(80, 3)
print(x_test.shape)     #(20, 3)

#2. Modelong
model = Sequential()
model.add(Dense(100, input_dim = 3)) #input 3개
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.3)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae )

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)