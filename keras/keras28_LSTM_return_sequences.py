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

print(x.shape)  #(13, 3)
print(y.shape)  #(13, )

x_pred = np.array([50,60,70])   # 목표 예상값 80
x_pred = x_pred.reshape(1, 3)

# 전처리
# LSTM에서 Minmaxscaler : 
# Minmaxscaler는 3차원을 처리할 수 없기 때문에 전처리를 한 후에 3차원으로 reshape를 한다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=66)  

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

print(x_train.shape)    #(11, 3)
print(x_test.shape)     #(2, 3)

print(x_train.shape[0]) # 11
print(x_train.shape[1]) # 3

# x_train = x_train.reshape(11, 3, 1)   
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

print(x_train.shape)    #(11, 3, 1)


x_test = x_test.reshape(2, 3, 1)   
x_pred = x_pred.reshape(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# return_sequences = True
model.add(LSTM(65, activation='relu', input_shape=(3,1), return_sequences = True))
model.add(LSTM(65, activation='relu'))
model.add(Dense(39))
model.add(Dense(39))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(1))

# model.add(LSTM(65, activation='relu', input_shape=(3,1), return_sequences = True))
# model.add(LSTM(65, activation='relu', return_sequences = True))
# model.add(LSTM(39))
# model.add(Dense(39))
# model.add(Dense(26))
# model.add(Dense(26))
# model.add(Dense(13))
# model.add(Dense(13))
# model.add(Dense(1))

model.summary()

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 65)             17420
_________________________________________________________________
lstm_1 (LSTM)                (None, 65)                34060
_________________________________________________________________
dense (Dense)                (None, 39)                2574
_________________________________________________________________
dense_1 (Dense)              (None, 39)                1560
_________________________________________________________________
dense_2 (Dense)              (None, 26)                1040
_________________________________________________________________
dense_3 (Dense)              (None, 13)                351
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 14
=================================================================
Total params: 57,019
Trainable params: 57,019
Non-trainable params: 0
_________________________________________________________________

* Param # : 4 X (input_dim + bias + output) X output
* 아웃풋은 하위레이어의 인풋, output shape의 맨 끝에 붙는다. (= 첫번째 노드의 아웃풋이 두번째 노드의 인풋이 된다. )
* 그냥 LSTM을 쓰면 3차원 인풋이 2차원 아웃풋으로 차원이 바뀌어서 나온다. >>> 그 다음 LSTM에 못 들어감 (error)
* return_sequences = True >>> 인풋했을 때의 차원을 그대로 유지해서 출력해준다. >>> 연산이 증가함
* lstm (LSTM) 레이어에서 Output Shape가 3차원이 됨 : (None, 3, 65)  >> 해당 레이어의 아웃풋인 65는 다음 레이어의 인풋이 된다.
* 4 * ( 65 + 1 + 65) * 65 = 34060

"""


#3. Compile.Train
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs= 1300, batch_size=5, validation_split=0.1, verbose=1, callbacks=[early_stopping])

#3. Evaluate, Predcit
loss, mae = model.evaluate(x_test,y_test,batch_size=3)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)

# LSTM 1번 사용했을 때 (405번)
# loss :  0.0009525410714559257
# mae :  0.030837297439575195
# y_pred :  [[79.593445]]

# LSTM 2번 사용했을 때  (564번)
# loss :  0.0012931515229865909
# mae :  0.03259921073913574
# y_pred :  [[80.73546]]

# LSTM 3번 사용했을 때 (707번)
# loss :  0.0014889168087393045
# mae :  0.03214216232299805
# y_pred :  [[78.06611]]
