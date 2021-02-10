# male, female 
# >> Imagegenerator, fit (넘파이로 저장해서 해야 함) 적용해서 완성

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization, AveragePooling2D, Activation
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
model.add(Conv2D(64, (2,2), padding='same', input_shape=(56, 56, 3), activation='relu'))
model.add(Conv2D(64,(2,2), activation='relu',  padding='same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
es = EarlyStopping(monitor='val_loss', patience=50, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=30, mode='min')
path = '../data/modelcheckpoint/k67_56_5_{val_loss:.3f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_data=(x_valid, y_valid), callbacks=[es, lr, cp])

loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.9621424078941345
# acc :  0.5428571701049805

# loss :  0.9163194894790649
# acc :  0.5571428537368774

# loss :  1.775903582572937
# acc :  0.5428571701049805

# loss :  0.6946509480476379
# acc :  0.48571428656578064

# loss :  2.7977030277252197
# acc :  0.5142857432365417

# loss :  1.3844975233078003
# acc :  0.5142857432365417

# loss :  0.6946845054626465
# acc :  0.48571428656578064