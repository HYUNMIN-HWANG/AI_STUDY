# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교

import numpy as np
from sklearn.datasets import load_iris

#1. DATA
dataset = load_iris()
x = dataset.data 
y = dataset.target 

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )
# print(x[:5])
# print(y)        # 나올 수 있는 경우의 수 3가지 : 0 , 1 , 2 (50개 씩) >>> 다중 분류 >>> 원핫인코딩해야 함

# x값 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=166)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)    # (120, 4, 1)
print(x_test.shape)     # (30, 4, 1)

# 다중 분류일 때, y값 전처리 One hot Encoding
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
# print(y_train)
# print(y_test)
print(y_train.shape)    # (120, 3) >>> output = 3
print(y_test.shape)     # (30, 3)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, padding='same',\
                input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=2,padding='same'))
model.add(Dropout(0.2))
model.add(MaxPool1D(pool_size=2))

model.add(Conv1D(filters=128, kernel_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))                  

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  

modelpath = '../data/modelcheckpoint/k54_5_iris_{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

model.fit(x_train, y_train, epochs=30, batch_size=5, validation_split=0.2, \
            verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
loss, acc  = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ", loss)
print("accuracy : ", acc)

print("y_test :",np.argmax(y_test[-5 : -1],axis=1))

y_predict = model.predict(x_test[-5:-1])
print("y_predict :", np.argmax(y_predict,axis=1))


# Dense
# loss :  0.12442982941865921
# accuracy :  0.9666666388511658
# mae :  0.06698165833950043
# y_data : [2 0 0 1]
# y_predict : [2 0 0 1]

# CNN
# loss :  0.06789936125278473
# accuracy :  1.0
# mae :  0.038619689643383026
# y_test : [1 0 2 1]
# y_predict : [1 0 2 1]

# Conv1D
# loss :  0.0947595164179802
# accuracy :  0.9666666388511658
# y_test : [1 0 2 1]
# y_predict : [1 0 2 1]
