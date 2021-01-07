# keras23_LSTM3.py
# 실습 LSTM 층을 2개 만들어라. (LSTM 1개와 성능비교하라)
# return_sequences = True
# reshape 하는 다른 방법 : x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

import numpy as np
#1. DATA
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# print(x.shape)  # (13, 3) --> (13, 3, 1) --> (3, 1)
# print(y.shape)  # (13, )

x_pred = np.array([50,60,70])   # 목표 예상값 80 # (3, )
x_pred = x_pred.reshape(1, 3)

# 전처리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=113)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# print(x_train.shape)    #(10, 3)
# print(x_test.shape)     #(3, 3)
# print(x_pred.shape)     #(1, 3)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_pred= x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)

# print(x_train.shape)    #(10, 3, 1)
# print(x_test.shape)     #(3, 3, 1)
# print(x_pred.shape)     #(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, input_shape=(3, 1), activation='relu', return_sequences = True))
model.add(LSTM(15, activation='relu'))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print(loss, mae)

y_pred = model.predict(x_pred)
print(y_pred)