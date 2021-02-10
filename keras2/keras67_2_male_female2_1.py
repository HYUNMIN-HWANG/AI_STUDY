# male, female 
# >> Imagegenerator, fit (넘파이로 저장해서 해야 함) 적용해서 완성

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, AveragePooling2D, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

#1. DATA
# npy load
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
x_valid = np.load('../data/image/gender/npy/keras67_valid_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
y_valid = np.load('../data/image/gender/npy/keras67_valid_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=42)

print(x_train.shape, x_valid.shape, x_test.shape)  # (1319, 56, 56, 3) (347, 56, 56, 3) (70, 56, 56, 3)
print(y_train.shape, y_valid.shape, y_test.shape)  # (1319,) (347,) (70,)

#2. Modeling
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', input_shape=(56, 56, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
es = EarlyStopping(monitor='val_loss', patience=50, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=30, mode='min')
path = '../data/modelcheckpoint/k67_{val_loss:.3f}_{epoch:02d}.h5'
cp = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_valid, y_valid), callbacks=[es, lr, cp])

loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.7593396902084351
# acc :  0.5539568066596985
