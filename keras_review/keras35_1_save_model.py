# 모델 저장하기 Model save

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(20, input_shape=(4,1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.save("./model/save_keras35.h5")