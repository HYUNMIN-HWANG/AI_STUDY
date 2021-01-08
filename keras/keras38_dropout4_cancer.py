# Dropout
# model.add(Dropout(0.2))
# Dropout 하기 전과 성능 비교

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

# print(x[:5])
# print(y)        # 0 or 1 >> classification (이진분류)

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (455, 2)
print(y_test.shape)     # (114, 2)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(30,)))  # input = 30
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(30, activation='relu'))   
model.add(Dropout(0.2)) 
model.add(Dense(2, activation='softmax'))                   # output = 2
            # output=2 : 결과 값이 나오는 숫자만큼 마지막 노드를 정한다.
            # 다중 분류 : softmax 
model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # 다중 분류 : categorical_crossentropy 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min') 
model.fit(x_train, y_train, epochs=4500, batch_size=10, validation_split=0.1, verbose=1,callbacks=[early_stopping])

#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test,batch_size=10)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)

y_predict = model.predict(x_test[-5:-1])

print("y_test_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)

print("result : ", np.argmax(y_predict,axis=1))

# loss :  0.4608054459095001
# accuracy :  0.9824561476707458
# mae :  0.020499642938375473
# y_test_data :
#  [[0. 1.]
#  [0. 1.]
#  [0. 1.]
#  [1. 0.]]
# y_predict :
#  [[3.7276200e-21 1.0000000e+00]
#  [3.8367564e-30 1.0000000e+00]
#  [1.9589033e-25 1.0000000e+00]
#  [1.0000000e+00 0.0000000e+00]]
# result :  [1 1 1 0]

# Dropout (성능 좋아짐)
# loss :  0.14030753076076508
# accuracy :  0.9649122953414917
# mae :  0.033343564718961716
# y_test_data :
#  [[0. 1.]
#  [0. 1.]
#  [0. 1.]
#  [1. 0.]]
# y_predict :
#  [[1.5221683e-05 9.9998474e-01]
#  [1.3640058e-13 1.0000000e+00]
#  [7.1588424e-03 9.9284112e-01]
#  [1.0000000e+00 0.0000000e+00]]
# result :  [1 1 1 0]