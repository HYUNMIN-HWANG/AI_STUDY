# PCA
# XGboost
# 실패

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold , cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from xgboost import XGBClassifier, plot_importance

######################################################

# 데이터 로드
train = pd.read_csv('../data/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

submission = pd.read_csv('../data/DACON_vision1/submission.csv')
print(submission.shape) # (20480, 2)

pred = pd.read_csv('../data/DACON_vision1/test.csv')
print(pred.shape)   # (20480, 786)

######################################################


#1. DATA
# x
x = train.drop(['id', 'digit', 'letter'], axis=1).values
x = x/255.0
# x = x.reshape(-1, 28, 28, 1)
print(x.shape)  # (2048, 28, 28, 1) >> (2048, 786)

# y
y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order) >> 0으로 초기화된 넘파이 배열 
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1
print(y.shape)  # (2048, 10)

# predict
x_pred = pred.drop(['id', 'letter'], axis=1).values
# x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255.0 
print(x_pred.shape) # (20480, 28, 28, 1)    >> (20480, 786)

# 컬럼 압축
# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)
# d = np.argmax(cumsum >= 0.95)+1
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)    # d :  98
pca = PCA(n_components=98)
x2 = pca.fit_transform(x)
x2_pred = pca.fit_transform(x_pred)

print(x2.shape)      # (2048, 98)
print(x2_pred.shape) # (20480, 98)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, shuffle=True, random_state=47)

######################################################

#2. Modeling
def modeling() : 
    model = Sequential()
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu',\
        input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    return model

kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=33)
    
for train_index, val_index in kf.split(x_train) :
    x_train, x_val = x2[train_index], x2[val_index]
    y_train, y_val = y[train_index], y[val_index]
    print(x_train.shape, x_val.shape)     # (1755, 98) (293, 98)
    print(y_train.shape, y_val.shape)     # (1755, 10) (293, 10)

    x_train = x_train.reshape(-1, 7 , 14 ,1)
    x_test = x_test.reshape(-1, 7 , 14 ,1)
    x_test = x_test.reshape(-1, 7 , 14 ,1)
    x2_pred = x2_pred.reshape(-1, 7, 14, 1)

    model = modeling()

    #3. Compile, Train
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    path = '../data/DACON_vision1/cp/0202_2_cp.hdf5'
    es = EarlyStopping(monitor='val_loss', patience=40, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=20, verbose=1, mode='min')
    cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

    model.fit(x_train, y_train, epochs = 1000, batch_size=14,\
        validation_split = 0.2, verbose=1, callbacks=[es, lr, cp])

    loss, acc = model.evaluate(x_test, y_test, batch_size=14)
    print("loss :", loss, "acc : ", acc)

    y_pred = model.predict(x2_pred)
    print(y_pred.shape) # 






# submission
submission['digit'] = np.argmax(y_pred, axis=1)
print(submission.head())

submission.to_csv('../data/DACON_vision1/baseline_0202_1.csv', index=False)

