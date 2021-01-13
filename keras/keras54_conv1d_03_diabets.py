# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교


import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

#1. DATA
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

# 전처리 과정

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],1)

print(x_train.shape)    # (282, 10, 1)
print(x_test.shape)     # (89, 10, 1)
print(x_validation.shape)   # (71, 10, 1)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

print(y_train.shape)    # (282, 1)
print(y_test.shape)     # (89, 1)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, padding='same', activation='relu', \
            input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(filters=100, kernel_size=2))
model.add(Dropout(0.2))
model.add(MaxPool1D(pool_size=2))

model.add(Conv1D(filters=200, kernel_size=2))
model.add(Dropout(0.4))
model.add(MaxPool1D(pool_size=2))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )

modelpath = '../data/modelcheckpoint/k54_3_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min') 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True, mode='min')

hist = model.fit(x_train, y_train, epochs=5000, batch_size=5, \
    validation_data=(x_validation, y_validation), verbose=1,callbacks=[es, cp])


#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
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

# Conv1D
# loss :  4501.46337890625
# mae :  53.776214599609375
# RMSE :  67.09294814400634
# R2 :  0.24410274295531076