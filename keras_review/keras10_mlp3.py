# 예측하고자 하는 y 값이 여러개일 때
# 임의의 예측값을 넣는다.
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array([range(100), range(101, 201), range(201, 301), range(301, 401), range(401, 501)])
y = np.array([range(211,311), range(512,612)])
print(x.shape) #(5, 100)
print(y.shape) #(2, 100)

x = np.transpose(x)
y = np.transpose(y)
print(x.shape) #(100, 5)
print(y.shape) #(100, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle=True,random_state=55)

x_pred2 = np.array([100, 200, 300, 400, 500]) 
print(x_pred2.shape) #(5,) ---> 스칼라

# x_pred2 = np.transpose(x_pred2) ----> #(5,) ----> 스칼라
x_pred2 = x_pred2.reshape(1, 5)
print(x_pred2) #[[100 200 300 400 500]]
print(x_pred2.shape)  #(1, 5) -----> 벡터(행렬)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) # output= 3개

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : : ', mae)

y_predict = model.predict(x_pred2)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_train))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
