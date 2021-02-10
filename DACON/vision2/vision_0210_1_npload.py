import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


######################################################
# File Load
# train = pd.read_csv('../data/DACON_vision2/dirty_mnist_2nd_answer.csv')
# print(train.shape)  # (50000, 27)

sub = pd.read_csv('../data/DACON_vision2/sample_submission.csv')
print(sub.shape)    # (5000, 27)

######################################################


#1. DATA
x = np.load('../data/DACON_vision2/npy/vision_x2.npy')
y = np.load('../data/DACON_vision2/npy/vision_y2.npy')
x_pred = np.load('../data/DACON_vision2/npy/vision_x_pred.npy')
print("<==complete load==>")

print(x.shape, y.shape, x_pred.shape) # (50000, 256, 256, 1) (50000, 26) (5000, 256, 256, 1)
x[x < 253] = 0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=47)
print(x_train.shape, x_test.shape, x_valid.shape)  # (32000, 256, 256, 1) (10000, 256, 256, 1) (8000, 256, 256, 1)
print(y_train.shape, y_test.shape, y_valid.shape)  # (32000, 26) (10000, 26) (8000, 26)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    fill_mode='nearest'
)
etc_datagen = ImageDataGenerator(rescale=1./255)

batch = 16

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch, seed=2021)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch, seed=2021)
valid_generator = etc_datagen.flow(x_valid, y_valid)
pred_generator = etc_datagen.flow(x_pred)

#2. Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', input_shape=(256, 256, 1), activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dense(26, activation='softmax'))

#3. Compile, Train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=None), metrics=['acc'])
hist = model.fit_generator(train_generator, epochs=30, \
    steps_per_epoch = len(x_train) // batch , validation_data=valid_generator, callbacks=[es, lr])

#4. Evaluate, Predict
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)  

# loss :  1269.6282958984375
# acc :  0.0364999994635582

y_pred = model.predict(pred_generator)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred < 0.5] = 0

print(y_pred.shape) # (5000, 26)

sub.iloc[:,1:] = y_pred

sub.to_csv('../data/DACON_vision2/sub_0210_1.csv', index=False)
print(sub.head())

# sub 제출안함
# score 	
