import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import PIL.Image as pilimg

######################################################
# File Load
train = pd.read_csv('../data/DACON_vision2/dirty_mnist_2nd_answer.csv')
print(train.shape)  # (50000, 27)

sub = pd.read_csv('../data/DACON_vision2/sample_submission.csv')
print(sub.shape)    # (5000, 27)

######################################################

#1. DATA

#### train
df_x = []

for i in range(0,50000):
    if i < 10 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/0000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/000' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/00' + str(i) + '.png'
    elif i >= 1000 and i < 10000 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/0' + str(i) + '.png'
    else : 
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_x.append(pix)

x = pd.concat(df_x)
x = x.values
print(x.shape)       # (12800000, 256) >>> (50000, 256, 256, 1)
x = x.reshape(50000, 256, 256, 1)
x = x/255
x = x.astype('float32')

y = train.iloc[:,1:]
print(y.shape)    # (50000, 27)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)

#### pred
df_pred = []

for i in range(0,5000):
    if i < 10 :
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/5000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/500' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/50' + str(i) + '.png'
    else : 
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/5' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pred.append(pix)

x_pred = pd.concat(df_pred)
x_pred = x_pred.values
print(x_pred.shape)       # (1280000, 256) >>> (5000, 256, 256, 1)
x_pred = x_pred.reshape(5000, 256, 256, 1)
x_pred = x_pred/255
x_pred = x_pred.astype('float32')

#2. Modeling
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same',input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(3,3))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(3,3))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization()) 
model.add(AveragePooling2D(2,2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization()) 
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))
model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=28, epochs=5, validation_split=0.2, verbose=1)

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=28)
print("loss : ", loss)
print("acc : ", acc)

# loss :  88.31974792480469
# acc :  0.027799999341368675

y_pred = model.predict(x_pred)
print(y_pred.shape) # (5000, 26)
print(y_pred)

# result = pd.DataFrame(y_pred)
# result = y_pred.to_numpy()
sub.iloc[:,1:] = y_pred

sub.to_csv('../data/DACON_vision2/0208_1.csv', index=False)

# sub_0208_2.csv
