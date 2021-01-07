# 저장한 모델 불러오기
# 실제 데이터를 넣어서 저장한 모델을 돌린다.

import numpy as np
a = np.array(range(1, 11))
size = 5

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
======================
        X     |  Y
======================
[[ 1  2  3  4 | 5]
 [ 2  3  4  5 | 6]
 [ 3  4  5  6 | 7]
 [ 4  5  6  7 | 8]
 [ 5  6  7  8 | 9]
 [ 6  7  8  9 | 10]]
'''

#1. DATA

x = dataset[:,:4] # [0:6,0:4]
# print(x) 
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
y = dataset[:,-1:] # [0:6,4:], [:, -1:]
# print(y)
# [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]

# x_pred = np.array([7,8,9,10])
x_pred = dataset[-1:,1:]

# print(x.shape)  # (6, 4) -> (6, 4, 1) -> (4, 1)   
# print(y.shape)  # (6, 1)

x = x.reshape(6, 4, 1)
x_pred = x_pred.reshape(1, 4, 1)

#2. Modeling
# 저장한 모델 불러오기
from tensorflow.keras.models import load_model
model = load_model ('./model/save_keras35.h5')   # input_shape = (4, 1)

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 200)               161600
_________________________________________________________________
dense (Dense)                (None, 100)               20100
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_2 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_3 (Dense)              (None, 10)                210
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 187,991
Trainable params: 187,991
Non-trainable params: 0
_________________________________________________________________

'''

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=200, batch_size=1)

#4. Evaluate, Predcit
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)


y_pred = model.predict(x_pred)
print("예측값 : ", y_pred)

# loss :  0.0005014999187551439
# mae :  0.020488103851675987
# 예측값 :  [[10.937883]]