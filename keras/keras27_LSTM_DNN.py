# keras23_LSTM3_scale을 DNN(Dense) 코딩
# 결과치 비교
# DNN으로 23번 파일보다 loss를 좋게 만들 것

# 실습 LSTM (예측값이 80이 나오도록 코딩하여라)

import numpy as np
#1. DATA
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)  #(13, 3)
print(y.shape)  #(13, )
# x = x.reshape(13, 3, 1)   

x_pred = np.array([50,60,70])   # 목표 예상값 80
x_pred = x_pred.reshape(1, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=44)  

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(52, activation='relu',input_shape=(3,)))
model.add(Dense(39))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(1))
# model.summary()

#3. Compile.Train
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=36,mode='min')

model.fit(x_train, y_train, epochs=2600, batch_size=13, validation_split=0.1, verbose=1, callbacks=[early_stopping])

#3. Evaluate, Predcit
loss, mae = model.evaluate(x_test,y_test,batch_size=13)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)

# LSTM (405번)
# loss :  0.0009525410714559257
# mae :  0.030837297439575195
# y_pred :  [[79.593445]]

# Dense (2600번)
# loss :  0.00017886147543322295
# mae :  0.01158285140991211
# y_pred :  [[79.99856]]
