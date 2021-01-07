import numpy as np
import tensorflow as tf 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_validation = np.array([6,7,8]) 
y_validation = np.array([6,7,8]) 

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

#2. Modeling
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))
                   
#3. Compile & Train
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_data=(x_validation, y_validation))


#4. Evaluate & Predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)

result = model.predict(x_train)
print("predict : ", result )