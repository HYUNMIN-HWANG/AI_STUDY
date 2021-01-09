# CNN 으로 구성
# 2차원 데이터를 4차원으로 늘려서 하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
datasets = load_breast_cancer()

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

x_train = x_train.reshape(x_train.shape[0],5,6,1)
x_test = x_test.reshape(x_test.shape[0],5,6,1)

print(x_train.shape)    # (512, 5, 6, 1)
print(x_test.shape)     # (57, 5, 6, 1)


# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
# print(y_train.shape)    # (455, 2)
# print(y_test.shape)     # (114, 2)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2,2),padding='same',input_shape=(5, 6, 1)))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=30,kernel_size=(2,2),padding='same'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))                   # output = 2

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # 다중 분류 : categorical_crossentropy 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.1, \
            verbose=1,callbacks=[early_stopping])

#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)


print("y_test :", np.argmax(y_test[-5 : -1],axis=1))
# print("y_predict :\n", y_predict)

y_predict = model.predict(x_test[-5:-1])
print("result : ", np.argmax(y_predict,axis=1))


# Dense
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

# CNN
# loss :  0.05749168619513512
# accuracy :  0.9649122953414917
# mae :  0.08164088428020477
# y_test : [1 1 1 0]
# result :  [1 1 1 0]