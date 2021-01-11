import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, x_test), (y_train, y_test) = mnist.load_data()

# x > preprocessing
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.

# y > preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test) 

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(2,2),padding='same',\
    input_shape=(x_train.shape[1],x_train.shape[2],,x_train.shape[3])))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=16, kernel_size=(4,4), padding='same', strides=1))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.1))
model.add(Flatten())

model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '파일을 저장할 위치'

es = EarlyStopping(monitor='val_loss',patience=5,mode='max')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.2.\
                callbacks=[es,cp])

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_sizes=32)
print("loss : ", result[0])
print("accuracy : ", result[1])

y_predict = model.predict(x_test[:10])
print("y_pred[:10] :")  
print(np.argmax(y_predict,axis=1))

# 시각화
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun Gothic')

plt.figure(figsize=(10,6))

plt.subplt(2, 1, 1)

plt.plot(hist.history['loss'], marker='.', c='red',label='loss')
plt.plot(hist.history(['val_loss'], marker='.',c='blue',label='val_loss'))
plt.grid()

plt.title('손실비용')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplt(2, 1, 2)

plt.plot(hist.history['acc'], marker='.', c='red',label='acc')
plt.plot(hist.history(['val_acc'], marker='.',c='blue',label='val_acc'))
plt.grid()

plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()