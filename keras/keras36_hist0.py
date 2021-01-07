# hist

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1, 101))
size = 5

def split_x(seq, size) :
    aaa = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

#1. DATA
dataset = split_x(a, size)
print(dataset.shape)    # (96, 5)

x = dataset[:, 0:4]
y = dataset[:,-1]
print(x.shape, y.shape) # (96, 4) (96,)

x = x.reshape(x.shape[0], x.shape[1],1) 
print(x.shape)  #(96, 4, 1)

#2. Modeling
model = load_model('./model/save_keras35.h5')    # input (4,1)
model.add(Dense(5, name='keras1'))
model.add(Dense(1, name='keras2'))

#3. Compile, Train

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=10,mode='min')

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=1000, batch_size=32, verbose=1,\
    validation_split=0.2, callbacks=[es])

print(hist)                 # <tensorflow.python.keras.callbacks.History object at 0x000001F554E46370>
print(hist.history.keys())  # dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])
# loss의 이력을 보여준다.
# erlystopping해서 멈춘 개수와 동일하다. (Epoch 67/1000)
'''[2283.443359375, 2221.573974609375, 2140.438720703125, 2006.113037109375, 1778.6920166015625, 
    1423.0255126953125, 989.4209594726562, 606.77001953125, 593.5076293945312, 554.4003295898438, 
    338.3880920410156, 271.37945556640625, 284.31866455078125, 228.89337158203125, 116.77024841308594, 
    35.24776077270508, 8.142428398132324, 1.7875198125839233, 1.0782527923583984, 1.475688099861145, 
    0.9893389344215393, 0.7049955129623413, 0.7913714051246643, 0.38927218317985535, 0.3677188456058502, 
    0.14482304453849792, 0.14060939848423004, 0.10975797474384308, 0.03892700374126434, 0.07369750738143921, 
    0.018675534054636955, 0.030280306935310364, 0.018825529143214226, 0.01587994396686554, 0.014620441012084484, 
    0.014188245870172977, 0.021246811375021935, 0.014071078971028328, 0.016989680007100105, 0.011916584335267544, 
    0.012025129050016403, 0.009882001206278801, 0.009832127951085567, 0.005501883570104837, 0.0034138846676796675, 
    0.0042558591812849045, 0.0037768229376524687, 0.0023513417690992355, 0.003464129753410816, 0.00315844570286572, 
    0.0030633017886430025, 0.0040892926044762135, 0.0022393884137272835, 0.003976468928158283, 0.002543087350204587, 
    0.002636508084833622, 0.001964522758498788, 0.0024039894342422485, 0.0032159474212676287, 0.0028372728265821934, 
    0.0030371861066669226, 0.0031387268099933863, 0.0027568088844418526, 0.002703902078792453, 0.002052864758297801, 
    0.00282532861456275, 0.0037725933361798525]'''

# 분류 모델이 아니기 때문에 acc는 별로 쓸모가 없음
# 일반적으로 loss 보다 val_loss가 더 값이 크다. 왜? 검증하는 것이기 때문에 훈련시키는 것보다 성능이 떨어진다.

# 그래프 그리기
import matplotlib.pyplot as plt 

plt.plot(hist.history['loss'])          #  loss의 이력 순서대로 그래프를 그림
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')                 #  그래프의 이름
plt.ylabel('loss, acc')                 #  y축 라벨링
plt.xlabel('epochs')                    #  x축 라벨링
plt.legend(['train loss','val loss','train acc','val acc']) # ← 그래프 빈 공간에 선에 대한 설명을 해준다.
plt.show()
