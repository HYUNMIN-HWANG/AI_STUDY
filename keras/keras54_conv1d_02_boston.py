# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교

# CNN 으로 구성
# 2차원 데이터를 4차원으로 늘려서 하시오.

import numpy as np

from sklearn.datasets import load_boston 
dataset = load_boston()

#1. DATA

x = dataset.data
y = dataset.target

# 다 : 1 mlp 모델을 구성하시오

# print(x.shape)  # (506, 13) input = 13
# print(y.shape)  # (506, )   output = 1
print('==========================================')

# ********* 데이터 전처리 *********

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size = 0.8, \
                                                                shuffle = True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

# print(np.max(x_train), np.min(x_train)) # 최댓값 1.0 , 최솟값 0.0
# print(np.max(x_train[0]))               # max = 0.9999

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],1)

print(x_train.shape)      # (323, 13, 1)
print(x_test.shape)       # (102, 13, 1)
print(x_validation.shape) # (81, 13, 1)

y_train = y_train.reshape (y_train.shape[0],1)
y_test = y_test.reshape (y_test.shape[0],1)

print(y_train.shape)    # (323, 1)
print(y_test.shape)     # (102, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=65, kernel_size=2,strides=1,padding='same',input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(filters=39, kernel_size=2,padding='same'))
model.add(Dropout(0.1))
model.add(MaxPool1D(pool_size=2))

model.add(Flatten())
model.add(Dense(26, activation='relu'))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

modelpath = '../data/modelcheckpoint/k54_2_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

model.fit(x_train, y_train, epochs=3000, batch_size=1, validation_data=(x_validation, y_validation), \
            verbose=1, callbacks=[es, cp])

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
# print("보스턴 집 값 : \n", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_train) :
    return np.sqrt(mean_squared_error(y_test, y_train))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)

# Dense
# loss :  5.84281587600708
# mae :  1.7922717332839966
# RMSE :  2.4171915790096543
# R2 :  0.9300955994589993

# CNN
# loss :  5.814030170440674
# mae :  1.8186489343643188
# RMSE :  2.4112293140745975
# R2 :  0.9304400277059781

# Conv1D
# loss :  7.530049800872803
# mae :  2.068243980407715
# RMSE :  2.744092948545423
# R2 :  0.9099092906921298