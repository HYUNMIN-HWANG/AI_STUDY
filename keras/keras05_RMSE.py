from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

# np.array
# array() : 위, 아래 같은 것

#1. DATA
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])

x_pred = array([16,17,18]) # y_pred : data 만들 필요없다. (어치파 예측할 것)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) #데이터를 20% 를 쪼개서 사용하겠다. (=10개 중에서 2개만 검증하겠다.) - validation data를 따로 지정하지 않아도 됨

#4. evaluate
results = model.evaluate(x_test, y_test, batch_size=1) #loss = 'mse', metrics='mae' 값이 들어간다
print("mse, mae :", results)

y_predict = model.predict(x_test)
# print("y_pred: ", y_predict)

#참고 : np.sqrt(results[0]) : result에 나오는 첫 번째 값이 mse이므로 루트를 씌우면 RMSE와 같다.

# 사이킷런(sklearn) 설치
from sklearn.metrics import mean_squared_error #mse
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict)) #RMSE = mse에 루트를 씌운다.
print("RMSE :", RMSE(y_test, y_predict))

# print("mse :", mean_squared_error(y_test, y_predict))
print("mse :", mean_squared_error(y_predict,y_test))

# RMSE 를 0.1 이하로 낮추시오

