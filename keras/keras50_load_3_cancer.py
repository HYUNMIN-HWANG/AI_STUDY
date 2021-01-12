# 저장한 numpy 불러오기 : load numpy

import numpy as np
x_data = np.load('../data/npy/cancer_x.npy')
y_data = np.load('../data/npy/cancer_y.npy')

# print(x_data)
# print(y_data)
print(x_data.shape) # (569, 30)
print(y_data.shape) # (569,)

# =========================== 모델을 완성하시오 ===========================

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=55)

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
model.add(Conv2D(filters=60, kernel_size=(2,2),padding='same',input_shape=(5, 6, 1)))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters=40,kernel_size=(2,2),padding='same'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))                   # output = 2

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # 다중 분류 : categorical_crossentropy 
modelpath='../data/modelcheckpoint/k46_6_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=10, mode='min') 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.1, \
            verbose=1,callbacks=[es,cp])

#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)


print("y_test :", np.argmax(y_test[-5 : -1],axis=1))
# print("y_predict :\n", y_predict)

y_predict = model.predict(x_test[-5:-1])
print("result : ", np.argmax(y_predict,axis=1))


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
# loss :  0.05619863420724869
# accuracy :  0.9824561476707458
# mae :  0.20988914370536804
# y_test : [1 1 1 0]
# result :  [1 1 1 0]

# ModelCheckpoint
# loss :  0.08481032401323318
# accuracy :  0.9649122953414917
# mae :  0.03930218890309334
# y_test : [1 1 1 0]
# result :  [1 1 1 0]