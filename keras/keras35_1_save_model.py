# 모델 저장하기  model.save
# 모델을 재사용하기 위해서

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. Modeling
model = Sequential()
model.add(LSTM(200, input_shape=(4, 1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

"""
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

"""
# model.save("모델을 저장할 경로지정")
# . <-- 점의 의미 : 현재 작업하고 있는 폴더 위치
# .. <-- 밑의 단계의 폴더로 간다.
# h5 : 확장자
model.save("../data/h5/save_keras35.h5")       # 가능
model.save("..//data//h5//save_keras35_1.h5")   # 가능
# model.save("..\data\h5\save_keras35_2.h5")     # 가능. 단, \n ,\t 같은 예약문자가 있을 때는 \\ 두개를 사용해야 한다.
model.save("..\\data\\h5\\save_keras35_3.h5")   # 가능
