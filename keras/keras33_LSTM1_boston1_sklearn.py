# sklearn
# LSTM으로 모델링 (Dense 와 성능 비교, Dense를 이긴다.)
# 회귀모델# MinMaxScalar

import numpy as np

from sklearn.datasets import load_boston #보스턴 집값에 대한 데이터 셋을 교육용으로 제공하고 있다.

dataset = load_boston()

#1. DATA

x = dataset.data
y = dataset.target # target : x와 y 가 분리한다.

# 다 : 1 mlp 모델을 구성하시오

print('==========================================')


# ********* 데이터 전처리 ( MinMax ) *********

# [3] x_train 만 전처리 한다.
# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=133)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)


print(x_train.shape)    #(404, 13)
print(x_test.shape)     #(102, 13)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)  
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)     

print(x_train.shape)    # (404, 13, 1)
print(x_test.shape)     # (102, 13, 1)

print('==========================================')

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(52, activation='relu',input_shape=(13,1)))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=13, mode='min')
model.fit(x_train, y_train, epochs=390, batch_size=13, validation_split=0.2, verbose=1, callbacks=[early_stopping])
# model.fit(x_train, y_train, epochs=390, batch_size=26, validation_split=0.2, verbose=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
# print("보스턴 집 값 : \n", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_train) :
    return np.sqrt(mean_squared_error(y_test, y_train))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)



#Dense
# # loss :  9.968693733215332
# mae :  2.3229475021362305
# RMSE :  3.157323994913747
# R2 :  0.8807328968436965

# LSTM
# loss :  8.347933769226074
# mae :  2.096160411834717
# RMSE :  2.889279024476091
# R2 :  0.767616336028877