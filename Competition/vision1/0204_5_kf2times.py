# private 3등 코드
# kf 2번 (총 150번 돌려야 함)
# 결과 별로임

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
# print(train, test, sub)
# print(train['digit'].value_counts())    # 0부터 9까지

x = train.drop(['id', 'digit','letter'],1).values
x_pred = test.drop(['id','letter'],1).values

# plt.imshow(train2[100].reshape(28,28))
# plt.show()

x = x.reshape(-1,28,28,1)
x_pred = x_pred.reshape(-1,28,28,1)

# preprocess
x = x/255.0
x_pred = x_pred/255.0

# y
y = train['digit']

#  ImageDataGenerator >> 데이터 증폭 : 데이터 양을 늘림으로써 오버피팅을 해결할 수 있다.
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

'''
sample_data = train2[100].copy()
sample = expand_dims(sample_data,0)
# expand_dims : 차원을 확장시킨다.
sample_datagen = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)    #  flow : ImageDataGenerator 디버깅

plt.figure(figsize=(16,10))
for i in range(9) :
    plt.subplot(3, 3, i+1)
    sample_batch = sample_generator.next()
    sample_image = sample_batch[0]
    plt.imshow(sample_image.reshape(28, 28))
plt.show()
'''

# cross validation
skf1 = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
skf2 = StratifiedKFold(n_splits=15, random_state=42, shuffle=True)

#2. Modeling
# %%time

reLR = ReduceLROnPlateau(patience=100, verbose=1, factor=0.5)
es = EarlyStopping(patience=120, verbose=1)

val_loss_min = []
val_acc_max = []
result = 0
nth = 0

for train_index, test_index in skf1.split(x, y) :
    path = '../data/DACON_vision1/cp/0204_5_cp.hdf5'
    mc = ModelCheckpoint(path, save_best_only=True, verbose=1)

    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    print(x_train.shape, x_test.shape)  # (1843, 28, 28, 1) (205, 28, 28, 1)
    print(y_train.shape, y_test.shape)  # (1843,) (205,)

    for train_index, valid_index in skf2.split(x_train, y_train) :
        x_train = x[train_index]
        x_valid = x[valid_index]
        y_train = y[train_index]
        y_valid = y[valid_index]

        train_generator = idg.flow(x_train, y_train, batch_size=32)
        test_generator = idg2.flow(x_test, y_test, batch_size=32)
        valid_generator = idg2.flow(x_valid, y_valid)
        pred_generator = idg2.flow(x_pred, shuffle=False)

        print(x_train.shape, x_test.shape, x_valid.shape)  # (1896, 28, 28, 1) (52, 28, 28, 1) (100, 28, 28, 1)
        print(y_train.shape, y_test.shape, y_valid.shape)  # (1896,) (52,) (100,)

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
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=None), metrics=['acc'])
                                                                            # epsilon : 0으로 나눠지는 것을 피하기 위함
        learning_hist = model.fit_generator(train_generator, epochs=1000, validation_data=valid_generator, callbacks=[es, mc, reLR] )
        model.load_weights('../data/DACON_vision1/cp/0204_5_cp.hdf5')

        #4. Evaluate, Predict
        loss, acc = model.evaluate(test_generator)
        print("loss : ", loss)
        print("acc : ", acc)

        result += model.predict_generator(pred_generator, verbose=True)/150

        # save val_loss
        hist = pd.DataFrame(learning_hist.history)
        val_loss_min.append(hist['val_loss'].min())
        val_acc_max.append(hist['val_acc'].max())

        nth += 1
        print(nth, "번째 학습을 완료했습니다.")

        print("val_loss_min :", np.mean(val_loss_min))  # val_loss_mean : val_loss_min : 0.2608904118090868
        print("val_acc_max :", np.mean(val_acc_max))    # val_acc_max : val_acc_max : 0.9372335501511891
        model.summary()

sub['digit'] = result.argmax(1)
print(sub)
sub.to_csv('../data/DACON_vision1/0204_5_sub.csv', index=False)

# submission 0204_2_sub
# score 0.9411764706 
