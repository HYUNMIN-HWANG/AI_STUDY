# '행'이 다른 앙상블 모델에 대해 공부하라 
# 결론 >> 앙상블을 사용할 때 무조건 행의 크기를 맞춰줘야 한다. 

import numpy as np

#1. DATA
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12]])
x2 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y1 = np.array([4,5,6,7,8,9,10,11,12,13]) 
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

print(x1.shape) # (10, 3)
print(x2.shape) # (13, 3)
print(y1.shape) # (10, )
print(y2.shape) # (13, )
print(x1_predict.shape) # (3,)
print(x2_predict.shape) # (3,)

x1 = x1.reshape(x1.shape[0], x1.shape[1],1)
x2 = x2.reshape(x2.shape[0], x2.shape[1],1)

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate

# 모델 두 개 구성
# 1 Model
input1 = Input(shape=(3, 1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)

# 2 Model
input2 = Input(shape=(3, ))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)

# 모델 병합
merge1 = concatenate([dense1, dense2])
middle1 = Dense(20, activation='relu')(merge1)
middle1 = Dense(20, activation='relu')(middle1)

# 모델 분기 
output1 = Dense(30, activation='relu')(middle1)
output1 = Dense(1)(output1)

output2 = Dense(30, activation='relu')(middle1)
output1 = Dense(1)(output2)  

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])
model.summary()

#3. Compile. Train
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit([x1, x2],[y1, y2],epochs=12, batch_size=1)

'''
ValueError: Data cardinality is ambiguous:
  x sizes: 10, 13
  y sizes: 10, 13
Please provide data which shares the same first dimension.
'''

#4. Evaluate, Predict
loss, mae = model.evaluate([x1, x2],[y1, y2],batch_size=1)
y_pred1, y_pred2 = model.predict([x1_predict, x2_predict])