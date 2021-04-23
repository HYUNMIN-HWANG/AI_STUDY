# 코랩으로 돌렸음 (경로 다름)
# pca (cumsum > 0.9999) : 625열로 압축
# 결과 너~~~~~~~무 안 좋음 
# pca 버려

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.decomposition import PCA
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# 데이터 로드
train = pd.read_csv('/content/drive/My Drive/AIA/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

sub = pd.read_csv('/content/drive/My Drive/AIA/DACON_vision1/submission.csv')
print(sub.shape) # (20480, 2)

test = pd.read_csv('/content/drive/My Drive/AIA/DACON_vision1/test.csv')
print(test.shape)   # (20480, 786)

#1. DATA
# print(train, test, sub)

# print(train['digit'].value_counts())    # 0부터 9까지

x = train.drop(['id', 'digit','letter'],1)
x_pred = test.drop(['id','letter'],1)

x = x.values  # >>> x
x_pred = x_pred.values    # >>> x_pred


# x = x.reshape(-1,28,28,1)
# x_pred = x_pred.reshape(-1,28,28,1)

# y
y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order) >> 0으로 초기화된 넘파이 배열 
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1
# print(y.shape)  # (2048, 10)

# preprocess
x = x/255.0
x_pred = x_pred/255.0

# pca
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)
d = np.argmax(cumsum >= 0.9999)+1
# print("cumsum >= 0.9999", cumsum > 0.9999)
print("d : ", d)    # d : 611

# 611 >> 제곱근 나오도록 625
pca = PCA(n_components=625)
x2 = pca.fit_transform(x)
x2_pred = pca.fit_transform(x_pred)

print(x2.shape)      # (2048, 625)
print(x2_pred.shape) # (20480, 625)

#  ImageDataGenerator >> 데이터 증폭 : 데이터 양을 늘림으로써 오버피팅을 해결할 수 있다.
# idg = ImageDataGenerator(height_shift_range=(-3,3) ,width_shift_range=(-3,3))
idg = ImageDataGenerator(height_shift_range=(-1,1) ,width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# cross validation
# skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)
skf = KFold(n_splits=30, random_state=42, shuffle=True)
 
# # Conv2D 하기 위해 4차원으로 변환
x2 = x2.reshape(-1, 25 , 25 ,1)
x2_pred = x2_pred.reshape(-1, 25, 25, 1)

print(x2.shape)       # (2048, 25, 25, 1)
print(x2_pred.shape)  # (20480, 25 , 25, 1)

reLR = ReduceLROnPlateau(patience=100, verbose=1, factor=0.5)
es = EarlyStopping(patience=120, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, test_index in skf.split(x2, y) : # >>> x, y
    path = '/content/drive/My Drive/AIA/DACON_vision1/cp/0203_4_cp.hdf5'
    mc = ModelCheckpoint(path, save_best_only=True, verbose=1)

    x_train = x2[train_index]
    x_test = x2[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=47)

    train_generator = idg.flow(x_train, y_train, batch_size=32)
    test_generator = idg2.flow(x_test, y_test, batch_size=32)
    valid_generator = idg2.flow(x_valid, y_valid)
    pred_generator = idg2.flow(x2_pred, shuffle=False)

    print(x_train.shape, x_test.shape, x_valid.shape)  # (1880, 25, 25, 1) (69, 25, 25, 1) (99, 25, 25, 1)
    print(y_train.shape, y_test.shape, y_valid.shape)  # (1880, 10) (69, 10) (99, 10)

    #2. Modeling
    model = Sequential()

    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(25, 25,1), padding='same'))
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
    # model.add(MaxPooling2D(3,3))
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
    model.load_weights('/content/drive/My Drive/AIA/DACON_vision1/cp/0203_4_cp.hdf5')

    #4. Evaluate, Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)

    result += model.predict_generator(pred_generator, verbose=True)/30

    # save val_loss
    hist = pd.DataFrame(learning_hist.history)
    val_loss_min.append(hist['val_loss'].min())

    nth += 1
    print(nth, "번째 학습을 완료했습니다.")

    print(val_loss_min, np.mean(val_loss_min))  # val_loss_min 2.2567497809727985
    model.summary()

sub['digit'] = result.argmax(1)
print(sub)
sub.to_csv('/content/drive/My Drive/AIA/DACON_vision1/0203_4_private3.csv', index=False)

# xian submission 0203_3_pca
# score 	0.2156862745	