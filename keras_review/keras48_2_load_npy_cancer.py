#1. DATA

import numpy as np

x_data = np.load('../data/npy/cancer_x.npy')
y_data = np.load('../data/npy/cancer_y.npy')

print(x_data.shape) # (569, 30)
print(y_data.shape) # (569,)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, \
    train_size=0.8, shuffle=True, random_state=66)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],5,6,1)
x_test = x_test.reshape(x_test.shape[0],5,6,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input

input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
conv = Conv2D(filters=256,kernel_size=(2,2), padding='same',\
    activation='relu')(input)
drop = Dropout(0.3)(conv)
conv = Conv2D(filters=256,kernel_size=(2,2),activation='relu')(drop)
drop = Dropout(0.3)(conv)
pool = MaxPool2D(pool_size=(2,2))(drop)

conv2 = Conv2D(filters=128,kernel_size=(2,2),activation='relu')(pool)
drop2 = Dropout(0.2)(conv2)
conv2 = Conv2D(filters=128,kernel_size=(2,2),activation='relu')(drop)
drop2 = Dropout(0.3)(conv2)
pool2 = MaxPool2D(pool_size=(2,2))(drop2)

flat = Flatten()(pool2)
dense = Dense(128)(flat)
dense = Dense(64)(dense)
dense = Dense(64)(dense)
output = Dense(2, activation='sigmoid')(dense)

model = Model(inputs=input, outputs=output)
# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss',patience=20, mode='min')
modelpath='../data/modelcheckpoint/k48_2_testdata_{epoch:02d}_{val_loss:.4f}.h5'
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True,mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, \
        validation_split=0.2, callbacks=[es, cp], verbose=1)

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("result : ", result)

print("y_test : ", np.argmax(y_test[:5], axis=1))
y_pred = model.predict(x_test[:5])
print("y_pred : ", np.argmax(y_pred, axis=1))
