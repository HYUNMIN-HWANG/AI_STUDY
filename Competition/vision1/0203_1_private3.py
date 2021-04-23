# private 3등 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
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

train2 = train.drop(['id', 'digit','letter'],1)
test2 = test.drop(['id','letter'],1)

train2 = train2.values  # >>> x
test2 = test2.values    # >>> x_pred

# plt.imshow(train2[100].reshape(28,28))
# plt.show()

train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# preprocess
train2 = train2/255.0
test2 = test2/255.0

#  ImageDataGenerator >> 데이터 증폭 : 데이터 양을 늘림으로써 오버피팅을 해결할 수 있다.
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
# width_shift_range : 왼쪽 오른쪽으로 움직인다.
# height_shift_range : 위쪽 아래쪽으로 움직인다.
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
skf = StratifiedKFold(n_splits=20, random_state=42, shuffle=True)

#2. Modeling
# %%time

reLR = ReduceLROnPlateau(patience=100, verbose=1, factor=0.5)
es = EarlyStopping(patience=120, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(train2, train['digit']) : # >>> x, y
    path = '../data/DACON_vision1/cp/0203_1_cp.hdf5'
    mc = ModelCheckpoint(path, save_best_only=True, verbose=1)

    x_train = train2[train_index]
    x_valid = train2[valid_index]
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]

    train_generator = idg.flow(x_train, y_train, batch_size=32)
    valid_generator = idg2.flow(x_valid, y_valid)
    test_generator = idg2.flow(test2, shuffle=False)

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
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['acc'])
                                                                        # epsilon : 0으로 나눠지는 것을 피하기 위함
    learning_hist = model.fit_generator(train_generator, epochs=1000, validation_data=valid_generator, callbacks=[es, mc, reLR] )

    #4. Evaluate, Predict
    model.load_weights('../data/DACON_vision1/cp/0203_1_cp.hdf5')
    result += model.predict_generator(test_generator, verbose=True)/40

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)
    val_loss_min.append(hist['val_loss'].min())

    nth += 1
    print(nth, "번째 학습을 완료했습니다.")

    print(val_loss_min, np.mean(val_loss_min))
    model.summary()

sub['digit'] = result.argmax(1)
print(sub)
sub.to_csv('../data/DACON_vision1/0203_1_private3.csv',index=False)