#훈련하면서 검증할 수 있는 데이터 설정하기 : validation
#validation_data

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense


#1. Data 
# 데이터 구분은 개발자들이 판단해서 나눈다.
# 통상적으로 훈련 데이터가 60% 정도 차지한다. 
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

# x_train 이외의 컴퓨터가 데이터를 검증할 데이터를 추가한다. 훈련하면서 검증한다. > 성능 좋아짐
x_validation = np.array([6,7,8])  
y_validation = np.array([6,7,8])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

#2. Model
model = Sequential()

model.add(Dense(10000, input_dim=1, activation='relu'))
model.add(Dense(1000))     
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) #output = 1

#3. Compile, Train
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # 출력결과 : accuracy == 0.0 (분류, 완전히 구분되어 있을 때 사용)
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # 출력결과 : mse == loss
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # mae : 평균 절대 오차

model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_validation, y_validation)
) # 훈련하면서 검증한다.

#4. evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

# result = model.predict([9])
result = model.predict(x_train)

print("result : ", result)