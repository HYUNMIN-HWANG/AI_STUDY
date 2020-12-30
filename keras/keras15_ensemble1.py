# ensemble

import numpy as np

#1. DATA
x1 = np.array( [range(100), range(301,401), range(1,101)] )         #(3, 100)
y1 = np.array( [range(711, 811), range(1, 101), range(201, 301)] )  

x2 = np.array([range(101, 201), range(411,511),range(100,200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)   #(100, 3)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)
from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split (x1, y1, shuffle=False, train_size=0.8)

#2. Modeling

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# [Model 1]
input1 = Input(shape=(3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)
output1 = Dense(3)(dense1)

# [Model 2]
input2 = Input(shape=(3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
output2 = Dense(3)(dense2)

#3. Compile, Train


#4. Evaluate, Perdict