# RNN > GRU

#1. DATA
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

print("x.shape : ", x.shape)  #(4, 3)
print("y.shape : ", y.shape)  #(4, )

# LSTM을 사용해서 계산하기 위해서 x 모양을 변경  >> ***3차원 데이터로 만들어준다.***
# 행 / 렬 / 몇 개씩 자르는지 (batch_size / time_steps / input_dim)
# 1개 데이터 씩 잘라서 3개를 계산해서 y 결과값 낸다.
# 데이터 자체의 손실 없음, 내용 변경 없음
x = x.reshape(4, 3, 1)        
# print(x)                    # [[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]], [[4],[5] [6]]]


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1)))   # x : 행을 무시한 데이터를 인풋
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390           
_________________________________________________________________
dense (Dense)                (None, 20)                220
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 831
Trainable params: 831
Non-trainable params: 0
_________________________________________________________________
* GRU의 activation default는? : tanh (-1과 1 사이에 값이 있다.)
* GRU의 gate 개수 : 3개 >> reset gate, update gate, hidden state >> LSTM에서 cell state 제거
* 최종으로 나가는 activation : tanh
* param # : 3 X (input_dim + bias(1) + output +1) X output
    ** reset_after=True 이기 때문에 마지막에 +1이 추가된다. (잘 모르겠음,,,,)
"""


#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. Evaluate, Predict
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7])          # (3, )
x_pred = x_pred.reshape(1, 3, 1)    # LSTM에 쓸 수 있는 3차원 데이터로 만들어준다

result = model.predict(x_pred)
print(result)                       # 결과값 8 예상

# LSTM
# 0.0033997094724327326
# [[8.175328]]

# SimpleRNN
# 0.014181404374539852
# [[8.232526]]

# GRU
# 0.005947352387011051
# [[7.7653513]]
