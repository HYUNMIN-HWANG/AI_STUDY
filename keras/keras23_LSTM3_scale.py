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
x = x.reshape(13, 3, 1)   

# 전처리
# LSTM에서 Minmaxscaler : 
# Minmaxscaler는 3차원을 처리할 수 없기 때문에 전처리를 한 후에 3차원으로 reshape를 한다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)  

x_pred = np.array([60,50,70])   # 목표 예상값 80
x_pred = x_pred.reshape(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(52, activation='relu',input_shape=(3,1)))
model.add(Dense(39))
model.add(Dense(39))
model.add(Dense(26))
model.add(Dense(26))
model.add(Dense(13))
model.add(Dense(1))
# model.summary()

#3. Compile.Train
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=390, batch_size=6, validation_split=0.1, verbose=1)

#3. Evaluate, Predcit
loss, mae = model.evaluate(x_test,y_test,batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)


# loss :  0.06494425237178802
# mae :  0.16342894732952118
# y_pred :  [[79.98596]]