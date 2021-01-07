# sklearn
# LSTM으로 모델링 (Dense 와 성능 비교)
# 회귀모델

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

#1. DATA
x = dataset.data
y = dataset.target

# print(x[:5])
# print(y[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

# 전처리 과정

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=16)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=16)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

# print(x_train.shape)    # (282, 10)
# print(x_test.shape)     # (89, 10)
# print(x_validation.shape) # (71, 10)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],1)

print(x_train.shape)    # (282, 10, 1)
print(x_test.shape)     # (89, 10, 1)
print(x_validation.shape) # (71, 10, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(10,1), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 

model.fit(x_train, y_train, epochs=200, batch_size=10, \
    validation_data=(x_validation, y_validation), verbose=1,callbacks=[early_stopping] )

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
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
r2 = r2_score (y_test, y_predict)
print("R2 : ", r2)

# Dense
# loss :  2305.467529296875
# mae :  39.62618637084961
# RMSE :  48.01528725236817
# R2 :  0.5981309295781687

# LSTM
# loss :  4516.42333984375
# mae :  56.398712158203125
# RMSE :  67.20434305025407
# R2 :  0.21879039545467793