# LSTM + ensemble (열이 다른 모델)

import numpy as np

#1. DATA
x1 = np.array([[1,2],[2,3],[3,4],[4,5],
                [5,6],[6,7],[7,8],[8,9],
                [9,10],[10,11],
                [20,30],[30,40],[40,50]])
x2 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])

y1 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_pred = np.array([55,65])    # >> 결과 3개
x2_pred = np.array([65,75,85]) # >> 결과 1개

print(x1.shape)            # (13, 2)
print(x2.shape)            # (13, 3)
print(y1.shape)            # (13, 3)
print(y2.shape)            # (13, )
print(x1_pred.shape)       # (2,) 
print(x2_pred.shape)       # (3,) 

x1_pred = x1_pred.reshape(1, x1_pred.shape[0])
x2_pred = x2_pred.reshape(1, x2_pred.shape[0])

# preprocessing
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, train_size=0.8, shuffle=True,random_state=55)
y1_train, y1_test, y2_train, y2_test = train_test_split(y1, y2, train_size=0.8, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_pred = scaler.transform(x1_pred)

scaler.fit(x2_train)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_pred = scaler.transform(x2_pred)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1)
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1)
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1],1)
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1],1)
x1_pred = x1_pred.reshape(1,x1_pred.shape[1],1)    # (1, 2, 1)
x2_pred = x2_pred.reshape(1,x2_pred.shape[1],1)    # (1, 3, 1)

print(x1_train.shape) # (10, 2, 1)
print(x1_test.shape)  # (3, 2, 1)
print(x1_pred.shape)  # (1, 2, 1)

print(x2_train.shape) # (10, 3, 1)
print(x2_test.shape)  # (3, 3, 1)
print(x2_pred.shape)  # (1, 3, 1)
# print(x)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate

# model = Sequential()
# model.add(LSTM(10, input_shape=(3, 1), activation='relu'))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(1))

input1 = Input(shape=(2,1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(9, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)

input2 = Input(shape=(3,1))
dense2 = LSTM(10, activation='relu')(input2)
dense2 = Dense(9, activation='relu')(dense2)
dense2 = Dense(8, activation='relu')(dense2)

merge1 = concatenate([dense1, dense2])
middle1 = Dense(6)(merge1)
middle1 = Dense(6)(middle1)

output1 = Dense(3)(middle1)
output2 = Dense(1)(middle1)

model = Model(inputs=[input1,input2], outputs=[output1,output2])

model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5,mode='min')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=100, batch_size=2, validation_split=0.2, verbose=2, callbacks=[es])

#4. Evaluate, Predict
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test], batch_size=2)
print("loss : ", loss )

y1_pred, y2_pred = model.predict([x1_pred,x2_pred])
print("y_pred1 : ", y1_pred)
print("y_pred2 : ", y2_pred)
