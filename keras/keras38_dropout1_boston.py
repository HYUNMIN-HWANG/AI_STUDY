# Dropout
# model.add(Dropout(0.2))
# Dropout 하기 전과 성능 비교

import numpy as np

from sklearn.datasets import load_boston #보스턴 집값에 대한 데이터 셋을 교육용으로 제공하고 있다.

dataset = load_boston()

#1. DATA

x = dataset.data
y = dataset.target # target : x와 y 가 분리한다.

# 다 : 1 mlp 모델을 구성하시오

# print(x.shape)  # (506, 13) input = 13
# print(y.shape)  # (506, )   output = 1
print('==========================================')

# ********* 데이터 전처리 ( MinMax ) *********

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state=66)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

print(np.max(x), np.min(x)) # 최댓값 711.0, 최솟값 0.0      ----> 최댓값 1.0 , 최솟값 0.0
print(np.max(x[0]))         # max = 396.9

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, activation='relu',input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min') 
#                               loss를 모니터링한다. / 가장 낮은 loss가 나온 후 20번만 더 시행한 후, 그때까지도 최솟값 변화가 없다면 스탑 / min, max, auto
# 장점 : 효율이 좋다. 얼추 효율이 좋은 epochs 위치를 알 수 있다. 정지할 수 있다. epochs 조절이 가능하다.

model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data=(x_validation, y_validation), verbose=1, callbacks=[early_stopping])

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
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

# early_stopping (20)  : 211번 실행
# loss :  6.976583957672119
# mae :  2.0358965396881104
# RMSE :  2.6413223452327177
# R2 :  0.91653100556001

# Dropout (성능 조금 좋아짐)
# loss :  5.84281587600708
# mae :  1.7922717332839966
# RMSE :  2.4171915790096543
# R2 :  0.9300955994589993