# keras23_3 LSTM을 Conv1D로 완성할 것
# LSTM과 결과값 비교
# 예측값이 80이 나오도록 코딩하여라

import numpy as np
#1. DATA
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)  #(13, 3)
print(y.shape)  #(13, )

x_pred = np.array([50,60,70])   # 목표 예상값 80
x_pred = x_pred.reshape(1, 3)


# 전처리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=88)  

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)   
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)   
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

print(x_train.shape)    # (11, 3, 1)
print(x_test.shape)     # (2, 3, 1)
print(x_pred.shape)     # (1, 3, 1)

y_train = y_train.reshape (y_train.shape[0],1)
y_test = y_test.reshape (y_test.shape[0],1)

print(y_train.shape)    # (11, 1)
print(y_test.shape)     # (2, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten
model = Sequential()
model.add(Conv1D(filters=65, kernel_size=2, activation='relu',input_shape=(3,1)))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(65))
model.add(Dense(39))
model.add(Dense(39))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(1))
# model.summary()

#3. Compile.Train
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k54_1_lstm_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

model.fit(x_train, y_train, epochs= 2600, batch_size=3, validation_split=0.1, verbose=1, callbacks=[es,cp])

#3. Evaluate, Predcit
loss, mae = model.evaluate(x_test,y_test,batch_size=3)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)

# LSTM (405번)
# loss :  0.0009525410714559257
# mae :  0.030837297439575195
# y_pred :  [[79.593445]]

# Dense (2600번)
# loss :  0.0005297368043102324
# mae :  0.02149200439453125
# y_pred :  [[80.04368]]

# Conv1D
# loss :  9.269416809082031
# mae :  2.892726421356201
# y_pred :  [[85.445244]]