# binary classification model

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target 

# print(x.shape)  # (569, 30)
# print(y.shape)  # (569,)

# print(x[:5])
# print(y)

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, \
    train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. Modeling

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(30,))                         #input = 30
dense1 = Dense(30, activation='relu')(input1)
dense1 = Dense(30, activation='relu')(dense1)
dense1 = Dense(30, activation='relu')(dense1)
output1 = Dense(1, activation='sigmoid')(dense1)    #output = 1
model = Model(inputs = input1, outputs=output1)
# model.summary()

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=10, batch_size=5, \
    validation_split=0.2, verbose=1,callbacks=[early_stopping])

#4. Evaluate. Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=5)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_test[-5:-1])
print("y_predict : \n",y_pred )
print("y_predict_binary : \n", np.round(y_pred,0))