# Simple RNN

#1, DATA
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

x_pred = np.array([5, 6, 7])
x_pred = x_pred.reshape(1, 3)

# preporcessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, \
    shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

print(x_train.shape)    #(3, 3)
print(x_test.shape)     #(1, 3)

x_train = x_train.reshape(3, 3, 1)
x_test = x_test.reshape(1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(40, activation='relu', input_shape=(3, 1)))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=100, batch_size=1, \
    validation_split=0.1, callbacks=[early_stopping])

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print(y_pred)


# loss :  8.423040390014648
# mae :  2.902247428894043
# [[11.636937]]