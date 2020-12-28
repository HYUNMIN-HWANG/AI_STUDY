from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

# np.array
# array() : 위, 아래 같은 것

#1. DATA

x = np.array(range(1,101)) # 1 ~ 100
# x = np.array(range(100))    # 0 ~ 99
y = np.array(range(101, 201))

#리스트의 슬라이싱 (train : validation : test = 6 : 2 : 2)
x_train = x[:60] # 0번째부터 59번째까지 : 1 ~ 60
x_val = x[60:80] # 60번째부터 79번째까지 : 61 ~ 80
x_test = x[80:]  # 81 ~ 100

y_train = y[:60] # 0번째부터 59번째까지 : 1 ~ 60
y_val = y[60:80] # 60번째부터 79번째까지 : 61 ~ 80
y_test = y[80:]  # 81 ~ 100

#2. Modeling
model = Sequential()
model.add(Dense(500, input_dim=1, activation='relu'))
model.add(Dense(200))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) #데이터를 20% 를 쪼개서 사용하겠다. (=10개 중에서 2개만 검증하겠다.) - validation data를 따로 지정하지 않아도 됨

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

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)