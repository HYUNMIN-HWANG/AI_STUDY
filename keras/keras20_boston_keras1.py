# 2개의 파일을 만드시오.
# 1/ Early Stopping을 적용하지 않은 최고의 모델 (파라미터 튜닝 완벽하게)

import numpy as np

#1. DATA
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

# sklearn의 x와 y를 가져오는 방식이 다르다.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=66)

print(x_train.shape)    #(323, 13)
print(x_test.shape)     #(102, 13)
print(np.min(x_train), np.max(x_train)) # 0.0 ~ 711.0

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

#2. Modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(65, input_dim=13, activation='relu'))
model.add(Dense(65))
model.add(Dense(52))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train, epochs=520, batch_size=13, validation_data=(x_validation, y_validation),verbose=1)

#4. Evaluate, Predcit
loss, mae = model.evaluate(x_test, y_test, batch_size=13)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# loss :  9.641185760498047
# mae :  2.226606845855713
# RMSE :  3.1050262503944763
# R2 :  0.8841813757163622
