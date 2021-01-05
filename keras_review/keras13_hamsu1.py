from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

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

input1 = Input(shape=(5,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs )
model.summary() 


model = Sequential()
model.add(Dense(10, input_shape=(5,)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))
model.summary()

#3. Compile. Train



#4. Evaluate, Predict