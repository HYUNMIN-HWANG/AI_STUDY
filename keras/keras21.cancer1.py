# breask cancer data
# classification model (이진분류 모델)
# loss = 'binary_crossentropy', activation = 'sigmoid', metrics=['acc']

# 실습1. acc : 0.985 이상 올릴 것
# 실습2. 일부데이터의 원래 데이터와 y_predict 데이터 비교
# 실습3. 결과가 0과 1 둘 중 하나로 나오게 코딩 할 것

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
datasets = load_breast_cancer()

# print(datasets.DESCR)           # (569, 30)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape)  #(569, 30) , input_dim = 30
print(y.shape)  #(568, ) # 유방암에 걸렸는지 안 걸렸는지 , output = 1

# print(x[:5])
# print(y)        # 0 or 1 >> classification (이진분류)

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(30,)))
model.add(Dense(90, activation='relu'))
model.add(Dense(150, activation='relu'))    
model.add(Dense(90, activation='relu'))    
model.add(Dense(30, activation='relu'))    
model.add(Dense(30, activation='relu'))    
model.add(Dense(1, activation='sigmoid'))  # sigmoid : 마지막 결과값이 0과 1사이로 나온다.


#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # 이진탐색에서는 무조건 loss='binary_crossentropy'
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 
# # mse == mean_squared_error / acc == accuracy (풀 네임으로 적어도 된다.)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 
model.fit(x_train, y_train, epochs=2000, batch_size=10, validation_split=0.1, verbose=1,callbacks=[early_stopping])

#4. Evalutate Predcit
loss, acc = model.evaluate(x_test, y_test,batch_size=5 )
print("loss : ",loss)
print("accuracy : ", acc)

y_predict = model.predict(x_test[-5:-1])

y_binary = list()           # y_preedict 값을 0과 1로 변환한 데이터 저장할 리스트
for i in y_predict :
    if i >= 0.5 :
        y_binary.append(1)  # y_preedict >= 0.5 이면 1
    else : 
        y_binary.append(0)  # y_preedict < 0.5 이면 0

print("y_test_data : ", y_test[-5 : -1])
print("y_predict :\n", y_predict) 
# 결과값으로 원했던 0과 1이 아닌 소수가 나온다. > sigmoid는 0과 1사이의 값이 나오기 때문 > 원하는 결과값이 나올 수 있도록 정제해야 함
print("result :", y_binary)
# print(np.round(y_predict,0)) # 반올림

# loss :  0.125028595328331
# accuracy :  0.9824561476707458
# y_test_data :  [1 1 1 0]
# y_predict :
#  [[9.997799e-01]
#  [9.999999e-01]
#  [9.996482e-01]
#  [3.975321e-27]]
# result : [1, 1, 1, 0]