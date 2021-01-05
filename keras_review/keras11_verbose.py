# 훈련과정을 생략한다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. DATA
x = np.array ([range(100), range(101, 201), range(201, 301), range(301, 401), range(401, 501)])
y = np.array ([range(411, 511), range(511,611)])

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=55)

x_pred = np.array([100,200,300,400,500])
x_pred = x_pred.reshape(1,5)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. Compile, Tran
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_split=0.2,verbose=0)

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_pred)
print(y_predict)
