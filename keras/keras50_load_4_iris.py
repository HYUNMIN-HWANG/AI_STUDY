# 저장한 numpy 불러오기 : np.load

import numpy as np
x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

# print(x_data)
# print(y_data)
print(x_data.shape) # (150, 4)
print(y_data.shape) # (150,)

# =========================== 모델을 완성하시오 ===========================

# x값 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=166)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],2,2,1)
x_test = x_test.reshape(x_test.shape[0],2,2,1)

print(x_train.shape)    # (120, 2, 2, 1)
print(x_test.shape)     # (30, 2, 2, 1)

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
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2,2), padding='same',\
                input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=128, kernel_size=(2,2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(2,2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=16, kernel_size=(2,2),padding='same'))
model.add(Conv2D(filters=16, kernel_size=(2,2),padding='same'))

model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(3, activation='softmax'))                  

model.summary()

#3. Compile, Train

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])  # acc == accuracy

modelpath='../data/modelcheckpoint/k46_7_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=10, mode='min') 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, \
            verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
loss, acc, mae  = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ", loss)
print("accuracy : ", acc)
print("mae : ", mae)


print("y_test :",np.argmax(y_test[-5 : -1],axis=1))

y_predict = model.predict(x_test[-5:-1])
print("y_predict :", np.argmax(y_predict,axis=1))

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))  # 판 사이즈 (가로 10, 세로 6)

plt.subplot(2, 1, 1)    # plot : 도화지 하나에 그림을 그리겠다.
                        # 2행 1열 중 첫 번째
                        # 만약 (3, 1, 1) 이라면 세 개의 plot이 있어야 한다. (3, 1, 1) (3, 1, 2) (3, 1, 3)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

# plt.title('손실비용') # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)    # 2행 1열 중 두 번째
plt.plot(hist.history['acc'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_accuracy')
plt.grid()              # 모눈종이 격자위에 그리겠다.

# plt.title('정확도')   # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# CNN
# loss :  0.06789936125278473
# accuracy :  1.0
# mae :  0.038619689643383026
# y_test : [1 0 2 1]
# y_predict : [1 0 2 1]

# ModelCheckPoint
# loss :  0.1256469041109085
# accuracy :  0.8999999761581421
# mae :  0.048764582723379135
# y_test : [1 0 2 1]
# y_predict : [1 0 2 1]