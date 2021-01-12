# model.save

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)  # (60000,)        (10000,)

# x > preprocessing
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1) / 255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1) / 255.

# y > preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',\
    input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(Dropout(0.3))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))
# model.summary()
# (1) 모델링 하고 난 직후 model.save
model.save('../data/h5/k51_test_model_1.h5')

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath='../data/modelcheckpoint/k51_test_{epoch:02d}_{val_loss:.4f}.h5'
es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min')

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[es,cp])
# (2) 컴파일, 훈련한 후 model.save
model.save('../data/h5/k51_test_model_2.h5')

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])


# Graph
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'],marker='.',c='red',label='acc')
plt.plot(hist.history['val_acc'],marker='.',c='blue',label='val_acc')
plt.grid()
plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()