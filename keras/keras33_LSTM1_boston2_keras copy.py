# tensorflow.keras
# LSTM으로 모델링 (Dense 와 성능 비교)
# 회귀모델

import numpy as np

#1. DATA
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

# sklearn의 x와 y를 가져오는 방식이 다르다.
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size = 0.9, shuffle = True, random_state=114)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

# print(x_train.shape)    # (323, 13)
# print(x_test.shape)     # (102, 13)
# print(x_validation.shape) # (81, 13)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],1)

print(x_train.shape)    # (323, 13, 1)
print(x_test.shape)     # (102, 13, 1)
print(x_validation.shape) # (81, 13, 1)

#2. Modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(65, input_shape=(13,1), activation='relu'))
model.add(Dense(65))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(1))

model.summary()


#3. Compile, Train
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
ealy_stopping = EarlyStopping(monitor='loss',patience=8,mode='min')
model.fit(x_train, y_train, epochs=260, batch_size=13, validation_data=(x_validation, y_validation),verbose=1, callbacks=[ealy_stopping])

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


# Dense
# loss :  9.107584953308105
# mae :  2.0973618030548096
# RMSE :  3.017877734702501
# R2 :  0.8905914829316571

# LSTM
# loss :  29.320236206054688
# mae :  3.777653932571411
# RMSE :  5.4148164106283945
# R2 :  0.647778937637217