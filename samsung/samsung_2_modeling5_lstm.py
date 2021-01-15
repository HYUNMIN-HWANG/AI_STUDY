# 다음 날 삼성의 '종가'를 예측한다.
# 함수형

import numpy as np

# 전체 DATASET
x_train_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[0]
x_test_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[1]
x_val_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[2]
y_train_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[3]
y_test_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[4]
y_val_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[5]
x_pred_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[6]
print(x_train_sam.shape)        # (1530, 6, 6)
print(x_test_sam.shape)         # (479, 6, 6)
print(x_val_sam.shape)          # (383, 6, 6)
print(y_train_sam.shape)        # (1530, 1)
print(y_test_sam.shape)         # (479, 1)
print(x_pred_sam.shape)         # (1, 6, 6)

size = 6
col = 6

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

# input1 = samsung model

input1 = Input(shape=(x_train_sam.shape[1],x_train_sam.shape[2]))
lstm1 = LSTM(1024, activation='relu')(input1)
drop1 = Dropout(0.3)(lstm1)
dense1 = Dense(512, activation='relu')(drop1)
# drop1 = Dropout(0.2)(dense1)
dense1 = Dense(256, activation='relu')(drop1)
# drop1 = Dropout(0.2)(dense1)
dense1 = Dense(124, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
output1 = Dense(1)(dense1)

# concatenate

# output

model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
savepath = './samsung/samsung_l_day15_{epoch:02d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss',patience=50,mode='min')
cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit(x_train_sam, y_train_sam, epochs=2000, batch_size=32, \
    validation_data=(x_val_sam, y_val_sam),verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
result = model.evaluate(x_test_sam, y_test_sam, batch_size=32)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred_sam = model.predict(x_test_sam)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))

RMSE_sam = RMSE(y_test_sam, y_pred_sam)
print("RMSE : ", RMSE_sam)

r2_sam = r2_score(y_test_sam, y_pred_sam)
print("R2 : ", r2_sam)

predict = model.predict(x_pred_sam)
print("L_1월 15일 삼성주가 예측 : ", predict)

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


# loss :  1186119.875
# mae :  781.32177734375
# RMSE :  1186120.0
# R2 :  0.9928092866830521
# 1월 15일 삼성주가 예측 :  [[87189.91]]

# loss :  5914134.5
# mae :  1913.9967041015625
# RMSE :  5914134.0
# R2 :  0.9641462570448077
# L_1월 15일 삼성주가 예측 :  [[88025.38]]

# loss :  2759532.0
# mae :  1220.189453125
# RMSE :  2759532.2
# R2 :  0.9832706592159258
# L_1월 15일 삼성주가 예측 :  [[91124.4]]