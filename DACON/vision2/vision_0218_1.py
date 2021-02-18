import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


######################################################
# File Load
dataset = pd.read_csv('../data/DACON_vision2/dirty_mnist_2nd_answer.csv')
y_df =  dataset.iloc[:,:]
print(y_df.shape)  # (50000, 27)

sub = pd.read_csv('../data/DACON_vision2/sample_submission.csv')
print(sub.shape)    # (5000, 27)

######################################################

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.2,
    fill_mode='nearest'
)
etc_datagen = ImageDataGenerator(rescale=1./255)

def mymodel () :
    model = Sequential()
    model.add(Conv2D(64, (2,2), padding='same', input_shape=(50, 50, 1), activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(AveragePooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(AveragePooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dense(1, activation='sigmoid'))

    return model

#1. DATA
x = np.load('../data/DACON_vision2/npy/vision_x3.npy')
# y = np.load('../data/DACON_vision2/npy/vision_y3.npy')
x_pred = np.load('../data/DACON_vision2/npy/vision_x_pred3.npy')
print("<==complete load==>")

print(x.shape, x_pred.shape) # (50000, 50, 50, 1) (5000, 50, 50, 1)
x[x < 253] = 0

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

val_loss_list = []
val_acc_list = []

for alph in alphabet :  # 한 알파벳씩 검증
    print('<<<<<<<<<<<< ', alph, ' predict start >>>>>>>>>>>>')
    y = y_df.loc[:,alph]
    y_pred_list = []
    y_result = 0

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9, shuffle=True, random_state=42)
    print(x_train.shape, x_test.shape, x_valid.shape)  # (32000, 50, 50, 1) (10000, 50, 50, 1) (8000, 50, 50, 1)
    print(y_train.shape, y_test.shape, y_valid.shape)  # (32000,) (10000,) (8000,)

    batch = 25
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch, seed=2021)
    test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch, seed=2021)
    valid_generator = etc_datagen.flow(x_valid, y_valid)
    pred_generator = etc_datagen.flow(x_pred)

    #2. Modeling
    model = mymodel()

    #3. Compile, Train

    path = '../data/DACON_vision2/cp/vision_0218_1_{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')

    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(lr=0.01, epsilon=None), 
        metrics=['acc']
    )

    hist = model.fit_generator(
        train_generator, epochs=100,
        steps_per_epoch = len(x_train) // batch ,
        validation_data=valid_generator, callbacks=[es, lr, cp]
    )

    #4. Evaluate, Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)  

    val_loss_list.append(loss)
    val_acc_list.append(acc)

    # loss :  
    # acc :  

    y_pred = model.predict(pred_generator)
    # print(y_pred.shape) # (5000, 1)
    y_result= np.where(y_pred < 0.5, 0, 1)
    sub.loc[:,alph] = y_result  # 예측값을 sub 파일에 저장
    print(sub.head())

    print("*******", alph, " predict end *******")

print("mean loss : ", sum(val_loss_list)/26)
print("mean acc : ", sum(val_acc_list)/26)

sub.to_csv('../data/DACON_vision2/sub_0218_1.csv', index=False)
print(sub.head())

# sub
# score 
