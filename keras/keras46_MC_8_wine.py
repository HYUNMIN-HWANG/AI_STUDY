# Dense
# from tensorflow.keras.callbacks import ModelCheckpoint

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

print(x_train.shape)    # (160, 13)
print(x_test.shape)     # (18, 13)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (160, 3)
print(y_test.shape)     # (18, 3)

#2. Modeling
#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(65, activation='relu', input_shape=(13,)))  # input = 13
model.add(Dropout(0.2))
model.add(Dense(65, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(13, activation='relu'))      
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))                   # output = 3

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # 다중 분류 : categorical_crossentropy 

modelpath='../data/modelcheckpoint/k46_8_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=5, mode='min') 
cp = ModelCheckpoint(filepath=modelpath,monitor='val_loss',save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=45, batch_size=13, validation_split=0.1, verbose=1,callbacks=[es, cp])


#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test, batch_size=13)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)

# print(x_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])

print("y_test_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)

print("result : ", np.argmax(y_predict,axis=1))


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

# ModelcheckPoint
# loss :  0.03402642905712128
# accuracy :  1.0
# mae :  0.017438583076000214
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[9.9979609e-01 1.9989558e-04 4.0929986e-06]
#  [7.9381913e-05 9.9990880e-01 1.1777326e-05]
#  [1.5623758e-05 2.1364729e-06 9.9998224e-01]
#  [9.9958664e-01 4.0594055e-04 7.4002423e-06]]
# result :  [0 1 2 0]