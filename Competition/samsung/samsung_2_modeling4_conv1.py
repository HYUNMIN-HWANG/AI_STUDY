# 다음 날 삼성의 '종가'를 예측한다.

import numpy as np

# 전체 DATASET
x_train = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[0]
x_test = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[1]
x_validation = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[2]
y_train = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[3]
y_test = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[4]
y_validation = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[5]
x_pred = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[6]
print(x_train.shape)        # (1530, 6, 6)
print(x_test.shape)         # (479, 6, 6)
print(x_validation.shape)   # (383, 6, 6)
print(y_train.shape)        # (1530, 1)
print(y_test.shape)         # (479, 1)
print(x_pred.shape)         # (1, 6, 6)

size = 6
col = 6

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D

model = Sequential()
model.add(Conv1D(filters=64,kernel_size=6,padding='same',\
    input_shape=(x_train.shape[1], x_train.shape[2]),activation='relu'))
model.add(Conv1D(filters=64,kernel_size=6,activation='relu',padding='same'))
model.add(Conv1D(filters=64,kernel_size=6,activation='relu',padding='same'))
# model.add(MaxPool1D(pool_size=2))

model.add(Conv1D(filters=128,kernel_size=6,activation='relu',padding='same'))
model.add(Conv1D(filters=128,kernel_size=6,activation='relu',padding='same'))

model.add(Conv1D(filters=256,kernel_size=6,activation='relu',padding='same'))
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
savepath = './samsung/samsung_c_day15_{epoch:02d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss',patience=30,mode='min')
cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=2000, batch_size=size, \
    validation_data=(x_validation, y_validation),verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=size)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

predict = model.predict(x_pred)
print("C_1월 15일 삼성주가 예측 : ", predict)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(25,15))  # 판 사이즈

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')   # label=' ' >> legend에서 설정한 위치에 라벨이 표시된다.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')   # loc 를 명시하지 않으면 그래프가 비어있는 지역에 자동으로 위치한다.

plt.show()

# Conv1D _

# loss :  736934.625
# mae :  633.7678833007812
# RMSE :  736933.44
# R2 :  0.9957099765748376
# 1월 15일 삼성주가 예측 :  [[90295.29]]

# loss :  525005.375
# mae :  530.5425415039062
# RMSE :  525004.44
# R2 :  0.9969437112748942
# C_1월 15일 삼성주가 예측 :  [[88172.336]]

# loss :  1391798.625
# mae :  892.0252075195312
# RMSE :  1391787.5
# R2 :  0.9918977737497564
# C_1월 15일 삼성주가 예측 :  [[91820.28]]

# loss :  1651826.875
# mae :  1018.89794921875
# RMSE :  1651824.9
# R2 :  0.9903839786914785
# C_1월 15일 삼성주가 예측 :  [[92654.016]]

# loss :  525730.8125
# mae :  516.55078125
# RMSE :  525728.75
# R2 :  0.9969394949266662
# C_1월 15일 삼성주가 예측 :  [[88027.28]]

# loss :  1023855.125
# mae :  773.5408935546875
# RMSE :  1023860.3
# R2 :  0.993792967097308
# C_1월 15일 삼성주가 예측 :  [[91634.03]]