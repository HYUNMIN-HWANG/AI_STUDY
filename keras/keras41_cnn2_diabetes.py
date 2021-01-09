# CNN 으로 구성
# 2차원 데이터를 4차원으로 늘려서 하시오.

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

#1. DATA
x = dataset.data
y = dataset.target

# print(x[:5])
# print(y[:10])

# print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

# print(np.max(x), np.min(y))     # 0.198787989657293 25.0  ---> 전처리 해야 함
# print(np.max(x), np.min(x))     # 0.198787989657293 -0.137767225690012
# print(dataset.feature_names)    # 10 column
                                  # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# 전처리 과정
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=55)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

x_train = x_train.reshape(x_train.shape[0],2,5,1)   
x_test = x_test.reshape(x_test.shape[0],2,5,1)     
x_validation = x_validation.reshape(x_validation.shape[0],2,5,1)     

# print(x_train.shape)    # (282, 2, 5, 1)
# print(x_test.shape)     # (89, 2, 5, 1)

# y_train = y_train.reshape(y_train.shape[0],1)
# y_test = y_test.reshape(y_test.shape[0], 1)

# print(y_train.shape)    # (282, 1)
# print(y_test.shape)     # (89, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=60,kernel_size=(2,2),padding='same',input_shape=(2,5,1)))
# model.add(MaxPool2D(pool_size=(1,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=40,kernel_size=(2,2),padding='same'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=30,kernel_size=(2,2),padding='same'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=11, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=2, validation_data=(x_validation, y_validation),\
             verbose=1,callbacks=[early_stopping] )

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score (y_test, y_predict)
print("R2 : ", r2)

# Dense
# loss :  2244.21240234375
# mae :  38.237037658691406
# RMSE :  47.373117107794975
# R2 :  0.49755893626202496

# CNN
# loss :  2562.849609375
# mae :  41.3016242980957
# RMSE :  50.62459321031188
# R2 :  0.5413657544942061
