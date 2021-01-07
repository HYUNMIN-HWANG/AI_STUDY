# LSTM 두 개가 들어간 ensemble 다 : 1 모델 (예측값을 85 근사치로 만들어라)

import numpy as np

#1. DATA
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

# print(x1.shape) #(13, 3)
# print(x2.shape) #(13, 3)
# print(y.shape)  #(13, )
# print(x1_predict.shape) # (3,)
# print(x2_predict.shape) # (3,)

x1_predict = x1_predict.reshape(1,3)
x2_predict = x2_predict.reshape(1,3)

# 전처리
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,\
   train_size=0.9, random_state=133)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
scaler.fit(x2_train)
x1_train = scaler.transform(x1_train)
x2_train = scaler.transform(x2_train)
x1_test = scaler.transform(x1_test)
x2_test = scaler.transform(x2_test)
x1_predict = scaler.transform(x1_predict)
x2_predict = scaler.transform(x2_predict)

print(x1_train.shape)       # (11, 3)
print(x1_test.shape)        # (2, 3)
print(x1_predict.shape)     # (1, 3)

# x >> 3 dimention

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1)
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1],1)
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1)
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1],1)
x1_predict = x1_predict.reshape(x1_predict.shape[0],x1_predict.shape[1],1)
x2_predict = x2_predict.reshape(x2_predict.shape[0],x2_predict.shape[1],1)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate

# 모델 두 개 구성
# 1 Model
input1 = Input(shape=(3, 1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)

# 2 Model
input2 = Input(shape=(3, 1))
dense2 = LSTM(10, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)

# 모델 병합
merge1 = concatenate([dense1, dense2])
middle1 = Dense(20, activation='relu')(merge1)
middle1 = Dense(20, activation='relu')(middle1)

# 모델 분기 (한 개만 있으면 됨)
output1 = Dense(30, activation='relu')(middle1)
output1 = Dense(1)(output1)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=output1)
# model.summary()

#3. Compile, Train
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit([x1_train, x2_train],y_train, epochs=100, batch_size=1)

#4. Evaluate, Predict
loss, mae = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict([x1_predict, x2_predict])
print(y_pred)

