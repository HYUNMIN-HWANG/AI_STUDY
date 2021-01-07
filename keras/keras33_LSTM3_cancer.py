# sklearn
# LSTM으로 모델링 (Dense 와 성능 비교)
# 이진분류

# loss = 'binary_crossentropy', activation = 'sigmoid', metrics=['acc']

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
datasets = load_breast_cancer()

# print(datasets.DESCR)           # (569, 30)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape)  #(569, 30) , input_dim = 30
# print(y.shape)  #(568, ) # 유방암에 걸렸는지 안 걸렸는지 , output = 1

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=13)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)    # (512, 30)
# print(x_test.shape)     # (57, 30)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)    # (512, 30, 1)
print(x_test.shape)     # (57, 30, 1)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(30,1)))
model.add(Dense(90, activation='relu'))
model.add(Dense(150, activation='relu'))    
model.add(Dense(90, activation='relu'))    
model.add(Dense(30, activation='relu'))    
model.add(Dense(1, activation='sigmoid'))  # sigmoid : 마지막 결과값이 0과 1사이로 나온다.

model.summary()

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=1,callbacks=[early_stopping])

#4. Evalutate Predcit
loss, acc = model.evaluate(x_test, y_test,batch_size=5 )
print("loss : ",loss)
print("accuracy : ", acc)

y_predict = model.predict(x_test[-5:-1])

print("y_test_data : ", y_test[-5:-1])
print("y_predict :\n", y_predict) 
# 결과값으로 원했던 0과 1이 아닌 소수가 나온다. > sigmoid는 0과 1사이의 값이 나오기 때문 > 원하는 결과값이 나올 수 있도록 정제해야 함

print(np.where(y_predict> 0.5, 1, 0))

# Dense
# loss :  0.125028595328331
# accuracy :  0.9824561476707458
# y_test_data :  [1 1 1 0]
# y_predict :
#  [[9.997799e-01]
#  [9.999999e-01]
#  [9.996482e-01]
#  [3.975321e-27]]
# result : [1, 1, 1, 0]

# LSTM
# loss :  0.06509558856487274
# accuracy :  0.9824561476707458
# y_test_data :  [0 1 0 0]
# y_predict :
#  [[1.9285475e-03]
#  [9.9100339e-01]
#  [6.2880666e-11]
#  [1.9475270e-02]]
# [[0]
#  [1]
#  [0]
#  [0]]