# private 2등 코드
# 노이즈 제
# 모르겠다

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import cv2
import gc
from keras import backend as bek

######################################################

# 데이터 로드
train = pd.read_csv('../data/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

sub = pd.read_csv('../data/DACON_vision1/submission.csv')
print(sub.shape) # (20480, 2)

test = pd.read_csv('../data/DACON_vision1/test.csv')
print(test.shape)   # (20480, 786)

######################################################

#1. DATA

# x_train
# print(train, test, sub)
# print(train['digit'].value_counts())    # 0부터 9까지
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)

x_train = np.where((x_train<=20)&(x_train!=0) ,0.,x_train)

x_train = x_train/255
x_train = x_train.astype('float32')

train_224=np.zeros([2048,300,300,3],dtype=np.float32)
for i, s in enumerate(x_train):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(300,300),interpolation = cv2.INTER_CUBIC)
    del converted
    train_224[i] = resized
    del resized
    bek.clear_session()
    gc.collect()

print("\ntrain_224 <done>\n")

# y
y = train['digit']
y_224 = np.zeros((len(y), len(y.unique())))  # 총 행의수 , 10(0~9)
for i, digit in enumerate(y):
    y_224[i, digit] = 1

# x_pred
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = np.where((x_test<=20)&(x_test!=0) ,0.,x_test)
x_test = x_test/255
x_test = x_test.astype('float32')

test_224=np.zeros([20480,300,300,3],dtype=np.float32)

for i, s in enumerate(x_test):
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(converted,(300,300),interpolation = cv2.INTER_CUBIC)
    del converted
    test_224[i] = resized
    del resized

bek.clear_session()
gc.collect()

print("\ntest_224 <done>\n")


#  ImageDataGenerator >> 데이터 증폭 : 데이터 양을 늘림으로써 오버피팅을 해결할 수 있다.
idg = ImageDataGenerator(width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)
idg2 = ImageDataGenerator()

'''
sample_data = train2[115].copy()
sample = expand_dims(sample_data,0)
# expand_dims : 차원을 확장시킨다.
sample_datagen = ImageDataGenerator(width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        rotation_range = 10,
        validation_split=0.2)
sample_generator = sample_datagen.flow(sample, batch_size=1)    #  flow : ImageDataGenerator 디버깅

plt.figure(figsize=(16,10))
for i in range(9) :
    plt.subplot(3, 3, i+1)
    sample_batch = sample_generator.next()
    sample_image = sample_batch[0]
    plt.imshow(sample_image.reshape(28, 28), cmap='Greys_r')
plt.show()
'''

# cross validation
skf = StratifiedKFold(n_splits=60, random_state=42, shuffle=True)

#2. Modeling
# %%time

reLR = ReduceLROnPlateau(patience=100, verbose=1, factor=0.5)
es = EarlyStopping(patience=120, verbose=1)

val_loss_min = []
val_acc_max = []
result = 0
nth = 0

for train_index, test_index in skf.split(train_224, y_224) : # >>> x, y
    path = '../data/DACON_vision1/cp/0204_3_cp.hdf5'
    mc = ModelCheckpoint(path, save_best_only=True, verbose=1)

    x_train = train_224[train_index]
    x_test = train_224[test_index]
    y_train = y_224[train_index]
    y_test = y_224[test_index]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=47)

    train_generator = idg.flow(x_train, y_train, batch_size=16)
    test_generator = idg2.flow(x_test, y_test, batch_size=16)
    valid_generator = idg2.flow(x_valid, y_valid)
    pred_generator = idg2.flow(test_224, shuffle=False)

    print(x_train.shape, x_test.shape, x_valid.shape)  # (1912, 300, 300, 3) (35, 300, 300, 3) (101, 300, 300, 3)
    print(y_train.shape, y_test.shape, y_valid.shape)  # (1912,) (35,) (101,)

    #2. Modeling
    model = Sequential()

    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28, 28,1), padding='same'))
    model.add(BatchNormalization()) 
    # BatchNormalization >> 학습하는 동안 모델이 추정한 입력 데이터 분포의 평균과 분산으로 normalization을 하고자 하는 것
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(3,3))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    #3. Compile, Train
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['acc'])
                                                                        # epsilon : 0으로 나눠지는 것을 피하기 위함
    learning_hist = model.fit_generator(train_generator, epochs=1000, validation_data=valid_generator, callbacks=[es, mc, reLR] )
    model.load_weights('../data/DACON_vision1/cp/0204_3_cp.hdf5')

    #4. Evaluate, Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)

    result += model.predict_generator(pred_generator, verbose=True)/60

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)
    val_loss_min.append(hist['val_loss'].min())
    val_acc_max.append(hist['val_acc'].max())

    nth += 1
    print(nth, "번째 학습을 완료했습니다.")

    print("val_loss_min :", np.mean(val_loss_min))  # val_loss_mean : 
    print("val_acc_max :", np.mean(val_acc_max))    # val_acc_max :
    model.summary()

sub['digit'] = result.argmax(1)
print(sub)
sub.to_csv('../data/DACON_vision1/0204_3_sub.csv', index=False)

# submission 
# score 	
