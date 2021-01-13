# Dnn, LSTM, Conv2d 중 가장 좋은 결과와 비교

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()

# print(dataset.DESCR)
# print(dataset.feature_names)

# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', \
# 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', \
# 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

#1. DATA

x = dataset.data
y = dataset.target

# print(x)        # preprocessing 해야 함
# print(y)        # 0, 1, 2 >> 다중분류
# print(x.shape)  # (178, 13)
# print(y.shape)  # (178, )

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)    # (160, 13, 1)
print(x_test.shape)     # (18, 13, 1)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (160, 3)
print(y_test.shape)     # (18, 3)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten

model = Sequential()
model.add(Conv1D(filters=65, kernel_size=2, padding='same', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(filters=65, kernel_size=2))
model.add(Dropout(0.2))
model.add(MaxPool1D(pool_size=2))

model.add(Conv1D(filters=130, kernel_size=2))
model.add(Dropout(0.2))
model.add(MaxPool1D(pool_size=2))

model.add(Flatten())
model.add(Dense(130))
model.add(Dense(130))
model.add(Dense(65))
model.add(Dense(65))
model.add(Dense(3, activation='softmax'))                   # output = 3

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  # 다중 분류 : categorical_crossentropy 
modelpath = '../data/modelcheckpoint/k54_6_wine_{epoch:02d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=5, mode='min') 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')
model.fit(x_train, y_train, epochs=45, batch_size=13, validation_split=0.1, verbose=1,callbacks=[es, cp])

#4. Evalutate Predcit
loss, acc = model.evaluate(x_test, y_test, batch_size=13)
print("loss : ",loss)
print("accuracy : ", acc)

# print(x_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])

print("y_test_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)

print("result : ", np.argmax(y_predict,axis=1))

# Dense
# loss :  0.023476235568523407
# accuracy :  1.0
# mae :  0.013629034161567688
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[9.9991262e-01 5.4553703e-07 8.6745582e-05]
#  [4.8492066e-06 9.9999499e-01 7.3400358e-08]
#  [4.3661150e-04 2.8053637e-05 9.9953532e-01]
#  [9.9972695e-01 2.6720166e-04 5.8833466e-06]]
# result :  [0 1 2 0]

# Dropout (성능 유사함)
# loss :  0.028161056339740753
# accuracy :  1.0
# mae :  0.01674073189496994
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[9.9903333e-01 9.0184662e-04 6.4793065e-05]
#  [4.2299906e-04 9.9957305e-01 3.8741309e-06]
#  [2.3380478e-07 2.4176154e-07 9.9999952e-01]
#  [9.9871445e-01 1.2303371e-03 5.5225402e-05]]
# result :  [0 1 2 0]

# Conv1D
# loss :  0.018114082515239716
# accuracy :  1.0
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[1.0000000e+00 4.4714589e-08 5.0024904e-11]
#  [1.0208355e-09 1.0000000e+00 3.9787579e-08]
#  [1.8781523e-07 1.8774504e-05 9.9998105e-01]
#  [9.9999833e-01 1.6195713e-06 7.2466629e-12]]
# result :  [0 1 2 0]