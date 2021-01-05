# 여러 입력을 받는 경우
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA 
import numpy as np

x = np.array([[1,2,3,4,5,6,7,8,9,10],\
            [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) #(2, 10)
print(y.shape) #(10,)

# 행과 열의 위치를 바꾼다. (열이 중요하기 때문에, 중요한 것을 열로 이동)
x = np.transpose(x)
print(x.shape) #(10, 2)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=2)) # input 2개
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile. Train
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)

#4. Evaluate, Predict
loss, mae = model.evaluate(x,y)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x)
