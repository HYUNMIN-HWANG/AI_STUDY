from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

#1. DATA
x = np.array(range(1,101))
y = np.array(range(1,101))

#train, test, validation 데이터 구분을 train_test_split를 사용해서 나눈다. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

print(x_train)
print(x_train.shape) #(80,)
print(x_test.shape)  #(20,)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Trian
model.compile (loss='mse', optimizer='adam', metrics=['mae'])
model.fit (x_train, y_train, epochs=100)

#4. Evalutate, Predict
loss, mae = model.evaluate(x_test, y_test)
print(loss)
print(mae)

y_pred = model.predict(x_test)
print(y_pred)