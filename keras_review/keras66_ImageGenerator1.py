import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. DATA

# [1] 선언
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# [2] 데이터화

xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size = (150, 150),
    batch_size = 10,
    class_mode='binary',
    save_to_dir='../data/image/brain_generator/train'
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150),
    batch_size=10,
    class_mode='binary',
    save_to_dir='../data/image/brain_generator/test'
)
# Found 120 images belonging to 2 classes.

# print(xy_train)
# print(xy_train[0])  # x , y
print(xy_train[0][0].shape) # x (10, 150, 150, 3)
print(xy_train[0][1].shape) # y (10,)

#2. Modeling
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(150, 150,3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.02), metrics=['acc'])

history = model.fit_generator(
    xy_train, steps_per_epoch=160/10, epochs=5, validation_data=xy_test, validation_steps=4, callbacks=[es, lr]
)

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

print("val_loss : ", val_loss[-1])