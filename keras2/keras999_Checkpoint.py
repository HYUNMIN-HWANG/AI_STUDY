# Checkpoint


import tensorflow as tf
import numpy as np
from tensorflow.train import Checkpoint, CheckpointManager, latest_checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
import os
 
#1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

print(x.shape)
print(x.dtype) 
 
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
 
model = Sequential() 
model.add(Dense(5, input_dim=1, activation='linear')) 
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))
 
cp_path = './DACON/cp/training-{epoch:04d}.ckpt'    # 가중치를 저장할 경로를 지정한다.
cp_dir = os.path.dirname(cp_path)                   # 가중치가 저장되어 있는 폴더 루트
cp = ModelCheckpoint(filepath=cp_path, \            
    save_weights_only=True, verbose=0, period=5)    # callback에 들어갈 ModelCheckPoint

#3. 컴파일 & 훈련
model.compile(loss = 'mse', optimizer=Adam(0.01)) 
model.fit(x, y, epochs= 100, batch_size=1, callbacks=[cp])  

latest = latest_checkpoint(cp_dir)                  # 가중치가 저장된 곳에서 가장 최근에 저장된 가중치를 불러온다.
print(">>>>>> latest : ", latest)
# >>>>>> latest :  ./DACON/cp\training-0100.ckpt
model.load_weights(latest)                          # 최신 가중치를 load_weights 한다.

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1) 
print('loss : ', loss)
 
result = model.predict([4])
print('result : ', result)

# loss :  2.7942220981458377e-08
# result :  [[3.9996846]]
