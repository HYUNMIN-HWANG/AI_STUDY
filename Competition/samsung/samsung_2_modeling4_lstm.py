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
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(512, activation='relu', \
    input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
savepath = './samsung/samsung_l_day15_{epoch:02d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss',patience=40,mode='min')
cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=2000, batch_size=3, \
    validation_data=(x_validation, y_validation),verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=3)
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
print("1월 15일 삼성주가 예측 : ", predict)

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

# loss :  1357655.25
# mae :  841.1729736328125
# RMSE :  1357655.1
# R2 :  0.9917693749055779
# 1월 15일 삼성주가 예측 :  [[90419.53]]

# loss :  723341.8125
# mae :  587.0309448242188
# RMSE :  723342.3
# R2 :  0.9956148223251793
# 1월 15일 삼성주가 예측 :  [[87296.086]]