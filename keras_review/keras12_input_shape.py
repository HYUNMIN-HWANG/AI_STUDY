# 다차원의 데이터를 입력시키기 위해서 input_dim=5 대신 input_shape=(5,) 사용한다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array( [range(100), range(1, 101), range(101,201), range(201, 301), range(301, 401)] ) 
y = np.array([range(511,611), range(611,711)])

x = np.transpose(x)     
y = np.transpose(y)     

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) #3개 행 모두를 행을 기준으로 자른다. #random_state : 랜덤 난수 고정

x_pred2 = np.array([100, 1, 101, 201, 301])
x_pred2 = x_pred2.reshape(1, 5) # [[100, 1, 101, 201, 301]] # inpurt_dim = 5

#2. Modeling
model = Sequential()
# model.add(Dense(10, input_dim=5))
model.add(Dense(10, input_shape = (5,)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. Compile, Trian
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0)  # 첫 번째 컬럼에서 20%, 두 번쩨 컬럼에서 20%, y 컬럼에서 20% # 이때 batch_size=1는 (1,3)을 의미함

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test) 
