# matplotlib >> scatter 
# 머신러닝 : 히든레이어가 없음


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

#1. DATA
x = np.arange(1, 11)
y = np.array([1,2,4,3,5,5,7,9,8,11])

print("\n",x,"\n",y)
#  [ 1  2  3  4  5  6  7  8  9 10] 
#  [ 1  2  4  3  5  5  7  9  8 11]

#2. Modeling
model = Sequential()
model.add(Dense(1, input_shape=(1,)))   # input = 1
# model.add(Dense(10))
# model.add(Dense(10))                  # 머신러닝에는 히든레이어가 없다. > 연산량이 적다. > 속도가 빠르다.
# model.add(Dense(1))                   # output = 1

#3. Compile, Train
optimizer = RMSprop(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x, y, epochs=1000)


#4. Evaluate, Predict
y_pred = model.predict(x)

plt.scatter(x, y)   # scatter : 흩어지게 하다 >> 그래프 위에 데이터 마다 점을 찍겠다.
plt.plot(x, y_pred, color='red')    # weight
plt.show()

