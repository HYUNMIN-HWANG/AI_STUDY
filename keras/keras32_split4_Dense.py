"""
과제 및 실습 : Dense 모델을 구성하시오
전처리, earlystopping 등등 배운 거 다 넣는다.

데이터 1 ~ 100 / 6개씩 자른다.

     x            y
1, 2, 3, 4, 5     6
    ...
95,96,97,98,99    100

predict 를 만들 것 (5,5)
96,97,98,99,100  --> 101
    ...
100,101,102,103,104 ---> 105
예상 predict는 (101,102,103,104,105)
"""

import numpy  as np

a = np.array(range(1, 101))
size = 6

# Dense 모델을 구성하시오

def split_x(seq, size) :
    aaa = []  
    for i in range(len(seq) - size + 1) :       # range(len(seq) - size + 1) : 반복횟수(= 행의 개수), # size : 열의 개수
        subset = seq[i : (i+size)]
        aaa.append(subset)
    # print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)  
# print(dataset)

'''
dataset
============================
        X             |  Y
============================
[[  1   2   3   4   5   6]
 [  2   3   4   5   6   7]
 [  3   4   5   6   7   8]
 [  4   5   6   7   8   9]
    .   .   .   .   .   .
 [ 92  93  94  95  96  97]
 [ 93  94  95  96  97  98]
 [ 94  95  96  97  98  99]
 [ 95  96  97  98  99 100]]
'''

#1. DATA

x = dataset[:,:5] 
# print(x) 
# print(x.shape)      # (95, 5) -> (95, 5, 1) -> (5, 1)

y = dataset[:,-1:]  
# print(y)
# print(y.shape)      # (95, 1)

pred = np.array(range(96, 106))
size_pred = 6
dataset_pred = split_x(pred, size_pred)  # (6, 5)

x_pred = dataset_pred[:,:5] 
# print(x_pred)
# print(x_pred.shape) # (5, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=33)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

# print(x_train.shape)    # (76, 5)
# print(x_test.shape)     # (19, 5)
# print(x_pred.shape)     # (5, 5)

# x_train = x_train.reshape(76, 5, 1)
# x_test = x_test.reshape(19, 5, 1)
# x_pred = x_pred.reshape(5, 5, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(125, activation='relu', input_shape=(5,)))
# model.add(Dense(125))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=10, mode='min')
model.fit(x_train, y_train, epochs=2000, batch_size=5, validation_split=0.2, callbacks=[early_stopping])

#4. Evaluate, Predcit
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)


y_pred = model.predict(x_pred)
print("Predict data : \n", y_pred)

# LSTM
# loss :  0.32072097063064575
# mae :  0.41145458817481995
# Predict data : 
#  [[103.028854]
#  [104.32126 ]
#  [105.62515 ]
#  [106.94057 ]
#  [108.26753 ]

# Dense
# loss :  0.00011799063213402405
# mae :  0.008983762934803963
# Predict data : 
#  [[101.01618]
#  [102.01643]
#  [103.01668]
#  [104.01692]
#  [105.01717]]

# 시계열 데이터라고 해서 무조건 LSTM 이 좋은 것은 아니다.