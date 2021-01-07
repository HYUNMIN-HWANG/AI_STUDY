# hist

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense


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
dataset = split_x(a,size)
print(dataset.shape)    # (96, 5)

x = dataset[:,:4]
y = dataset[:,-1]
print(x.shape)  # (96, 4)
print(y.shape)  # (96, )

x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)  # (96, 4, 1)

#2. Modeling
from tensorflow.keras.models import load_model
model = load_model("./model/save_keras35.h5")
model.add(Dense(10, name='new1'))
model.add(Dense(1, name='new2'))

model.summary()

#3. Compile, train

from tensorflow.keras.callbacks import EarlyStopping
ex = EarlyStopping(monitor='loss', patience=10, mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['acc'])
hist = model.fit(x, y, epochs=100, batch_size=1, \
    verbose=1, validation_split=0.1, callbacks=[ex])

print(hist)
print(hist.history.keys())

print(hist.history['loss'])

# Graph
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['loss','val loss','acc','val acc'])
plt.show()
#4. Evaluate, Predict

