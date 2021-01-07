# ensemble
# model1, model2 -> concatenate -> model1 

import numpy as np

#1. DATA
x1 = np.array ([range(100), range(101, 201), range(201,301), range(301, 401), range(401, 501)]) #(5, 100)
x2 = np.array ([range(20,120), range(40, 140), range(60, 160), range(80, 180), range(100,200)]) #(5, 100)
y1 = np.array ([range(10,110), range(111, 211), range(211,311), range(311, 411), range(411, 511)]) #(5, 100)

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = \
    train_test_split(x1, x2, y1, train_size=0.8, shuffle=False)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# Model 1
input1 = Input(shape=(5,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10,activation='relu')(dense1)
dense1 = Dense(10,activation='relu')(dense1)

# Mode1 2
input2 = Input(shape=(5,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(10,activation='relu')(dense2)
dense2 = Dense(10,activation='relu')(dense2)

# model merge
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(middle1)

# model dense
output1 = Dense(20)(middle1)
output1 = Dense(20)(output1)
output1 = Dense(5)(output1)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = output1)
model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train,\
         epochs=50, batch_size=1, validation_split=0.2, verbose=0)

#4. Evaluate, Predict
loss = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)
print("loss : ", loss)

y_predict = model.predict([x1_test, x2_test])
print("y_predict : \n", y_predict)      

# RMSE

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y1_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ", r2)
