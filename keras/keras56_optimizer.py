# optimizer 6개, learning rate

import numpy as np

#1. DATA
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. Modeling
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. Compile, Train
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# Adam
# optimizer = Adam(lr=0.1)  # learning rate 0.1
# loss :  54417.67578125 결과물 :  [[434.08746]]
# optimizer = Adam(lr=0.01)  # learning rate 0.01
# loss :  5.925926524659453e-13 결과물 :  [[10.999998]]
# optimizer = Adam(lr=0.001)  # learning rate 0.001
# loss :  5.286438062475363e-13 결과물 :  [[11.]]
# optimizer = Adam(lr=0.0001)  # learning rate 0.0001
# loss :  7.050474093439618e-10 결과물 :  [[10.999944]] # epochs이 부족하기 때문

# Adadelta
# optimizer = Adadelta(lr=0.1)  # learning rate 0.1
# loss :  0.0004708416818175465 결과물 :  [[10.960004]]
# optimizer = Adadelta(lr=0.01)  # learning rate 0.01
# loss :  0.0003652046143542975 결과물 :  [[11.033621]]
# optimizer = Adadelta(lr=0.001)  # learning rate 0.001
# loss :  3.2144598960876465 결과물 :  [[7.7350907]]
# optimizer = Adadelta(lr=0.0001)  # learning rate 0.0001
# loss :  35.513702392578125 결과물 :  [[0.42588595]]

# Adamax
# optimizer = Adamax(lr=0.1)  # learning rate 0.1
# loss :  3.486897549009882e-05 결과물 :  [[10.99143]]
# optimizer = Adamax(lr=0.01)  # learning rate 0.01
# loss :  5.7059780511625746e-11 결과물 :  [[11.000004]]
# optimizer = Adamax(lr=0.001)  # learning rate 0.001
# loss :  3.7245275308350756e-08 결과물 :  [[11.000268]]
# optimizer = Adamax(lr=0.0001)  # learning rate 0.0001
# loss :  0.0017851043958216906 결과물 :  [[10.944207]]

# Adagrad
# optimizer = Adagrad(lr=0.1)  # learning rate 0.1
# loss :  18117610.0 결과물 :  [[-5979.133]]
# optimizer = Adagrad(lr=0.01)  # learning rate 0.01
# loss :  8.543948752048891e-06 결과물 :  [[10.997465]]
# optimizer = Adagrad(lr=0.001)  # learning rate 0.001
# loss :  6.386226232280023e-07 결과물 :  [[10.998324]]
# optimizer = Adagrad(lr=0.0001)  # learning rate 0.0001
# loss :  0.00424690218642354 결과물 :  [[10.921017]]

# RMSprop
# optimizer = RMSprop(lr=0.1)  # learning rate 0.1
# loss :  18019132637184.0 결과물 :  [[9027455.]]
# optimizer = RMSprop(lr=0.01)  # learning rate 0.01
# loss :  4.2985687255859375 결과물 :  [[7.47851]]
# optimizer = RMSprop(lr=0.001)  # learning rate 0.001
# loss :  0.026998335495591164 결과물 :  [[10.704612]]
# optimizer = RMSprop(lr=0.0001)  # learning rate 0.0001
# loss :  1.948445742527838e-06 결과물 :  [[10.998901]]

# SGD 
# optimizer = SGD(lr=0.1)  # learning rate 0.1
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr=0.01)  # learning rate 0.01
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr=0.001)  # learning rate 0.001
# loss :  1.1331735896846773e-11 결과물 :  [[11.000003]]
# optimizer = SGD(lr=0.0001)  # learning rate 0.0001
# loss :  0.0012878321576863527 결과물 :  [[10.96677]]

# Nadam
# optimizer = Nadam(lr=0.1)  # learning rate 0.1
# loss :  42760409186304.0 결과물 :  [[13623061.]]
# optimizer = Nadam(lr=0.01)  # learning rate 0.01
# loss :  1.30739860669353e-13 결과물 :  [[10.999999]]
optimizer = Nadam(lr=0.001)  # learning rate 0.001
# loss :  7.176481631177012e-13 결과물 :  [[11.000002]]
# optimizer = Nadam(lr=0.0001)  # learning rate 0.0001
# loss :  8.203258516914502e-07 결과물 :  [[10.998085]]


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4. Evaluate, Predict
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)