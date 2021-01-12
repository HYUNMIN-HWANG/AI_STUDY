# 저장한 numpy 불러오기 : np.load

import numpy as np
x_data = np.load('../data/npy/wine_x.npy')
y_data = np.load('../data/npy/wine_y.npy')

# print(x_data)
# print(y_data)
print(x_data.shape) # (178, 13)
print(y_data.shape) # (178,)

# =========================== 모델을 완성하시오 ===========================

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)    # (160, 13)
print(x_test.shape)     # (18, 13)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (160, 3)
print(y_test.shape)     # (18, 3)

#2. Modeling
#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(65, activation='relu', input_shape=(13,)))  # input = 13
model.add(Dropout(0.2))
model.add(Dense(65, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))      
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))                   # output = 3

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # 다중 분류 : categorical_crossentropy 

modelpath='../data/modelcheckpoint/k46_8_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=5, mode='min') 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=45, batch_size=13, validation_split=0.1, verbose=1,callbacks=[es, cp])


#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test, batch_size=13)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)

# print(x_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])

print("y_test_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)

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

# Dropout (성능 유사함)
# loss :  0.028161056339740753
# accuracy :  1.0
# mae :  0.01674073189496994
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[9.9903333e-01 9.0184662e-04 6.4793065e-05]
#  [4.2299906e-04 9.9957305e-01 3.8741309e-06]
#  [2.3380478e-07 2.4176154e-07 9.9999952e-01]
#  [9.9871445e-01 1.2303371e-03 5.5225402e-05]]
# result :  [0 1 2 0]

# ModelcheckPoint
# loss :  0.019184239208698273
# accuracy :  1.0
# mae :  0.012225303798913956
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[9.9996269e-01 3.6701418e-05 6.0222499e-07]
#  [7.4334024e-04 9.9759442e-01 1.6621642e-03]
#  [3.2257705e-04 1.7282944e-02 9.8239440e-01]
#  [9.9987376e-01 1.2526810e-04 1.0112942e-06]]
# result :  [0 1 2 0]