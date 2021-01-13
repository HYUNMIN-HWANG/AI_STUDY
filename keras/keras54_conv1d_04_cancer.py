# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교

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

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)    # (512, 30, 1)
print(x_test.shape)     # (57, 30, 1)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (512, 2)
print(y_test.shape)     # (57, 2)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=300, kernel_size=2,padding='same',input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.3))
model.add(Conv1D(filters=300,kernel_size=2,padding='same'))
model.add(Dropout(0.3))
model.add(MaxPool1D(pool_size=2))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))                   # output = 2

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # 다중 분류 : categorical_crossentropy 

modelpath = '../data/modelcheckpoint/k54_4_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss', save_best_only=True, mode='min')

model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.1, \
            verbose=1,callbacks=[es, cp])

#4. Evalutate Predcit
loss, acc = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ",loss)
print("accuracy : ", acc)


print("y_test :", np.argmax(y_test[-5 : -1],axis=1))
# print("y_predict :\n", y_predict)

y_predict = model.predict(x_test[-5:-1])
print("result : ", np.argmax(y_predict,axis=1))


# Dense
# loss :  0.14030753076076508
# accuracy :  0.9649122953414917
# mae :  0.033343564718961716
# y_test_data : [1 1 1 0]
# y_predict : [1 1 1 0]

# CNN
# loss :  0.05619863420724869
# accuracy :  0.9824561476707458
# mae :  0.20988914370536804
# y_test : [1 1 1 0]
# result :  [1 1 1 0]

# Conv1D
# loss :  0.06397397816181183
# accuracy :  0.9649122953414917
# y_test : [1 1 1 0]
# result :  [1 1 1 0]