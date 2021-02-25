# warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
# from keras.utils import np_utils
import cv2

import gc
from keras import backend as bek

########################################
train = pd.read_csv('../data/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

submission = pd.read_csv('../data/DACON_vision1/submission.csv')
print(submission.shape) # (20480, 2)

test = pd.read_csv('../data/DACON_vision1/test.csv')
print(test.shape)   # (20480, 786)
########################################

from sklearn.model_selection import train_test_split
# x
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 56, 56, 1)

x_train = np.where((x_train<=20)&(x_train!=0) ,0.,x_train)  # 특징이 낮은 것들은 모두 0으로 반환
x_train = np.where((x_train>=80) ,252.,x_train)  # 알파벳만 강조

x_train = x_train.astype('float32')
x_train224=np.zeros([2048,56,56,3],dtype=np.float32) # [2048,56,56,3] 의 검은색 배경을 만들어 준다.

for i, s in enumerate(x_train):
    print(i)
    converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
    resized = cv2.resize(converted,(56,56),interpolation = cv2.INTER_CUBIC) # (56, 56)으로 리사이즈
    blur =  cv2.GaussianBlur(resized, (5,5), cv2.BORDER_DEFAULT)
    threshold, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY) 
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    del converted
    x_train224[i] = thresh
    # cv2.imshow("resized_224", train_224[i])
    # cv2.waitKey(0)
    del thresh
    bek.clear_session()
    gc.collect()

x_train224 = x_train224/255

print(x_train224.shape)    # (2048, 56, 56, 3)

# y
y = train['letter']
y_train = np.zeros((len(y), len(y.unique())))  # 총 행의수 26

i = 0
for letter in y :
    # print(i, letter)
    # 문자열을 아스키 코드로 변환, A = 0 부터 시작하도록 지정
    asc =  ord(letter) - 65
    y_train[i, asc] = 1
    i += 1
print(y_train.shape)    # (2048, 26)

# y = A(0) case ; A인 경우만 모은다.
A_x = []
A_y = []

for i in range(2048) :
    idx = 0
    if y_train[i,0] == 1.0 :
        # A_x[idx] = x_train224[i]
        # A_y[idx] = y_train[i]
        A_x.append(x_train224[i])
        A_y.append(y_train[i])


# print(A_x)
# print(A_y)
# print(len(A_x)) # 72
# print(len(A_y)) #72

A_x = np.array(A_x)
A_y = np.array(A_y)
print(A_x.shape)    # (72, 56, 56, 3)
print(A_y.shape)    # (72, 26)



from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

datagen = ImageDataGenerator(   # 이미지 증폭
        width_shift_range=0.05,
        height_shift_range=0.05,
        rotation_range = 10,
        validation_split=0.2)

valgen = ImageDataGenerator()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def create_model() :    # Model
    
    effnet = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,
        input_shape=(56,56,3),
        classes=10,
        classifier_activation="softmax",
    )
    model = Sequential()
    model.add(effnet)


    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=RMSprop(lr=initial_learningrate),
                metrics=['accuracy'])
    return model

initial_learningrate=2e-3  # 가중치

from sklearn.model_selection import KFold

kf = KFold(n_splits=2, random_state=40)    # 50번 반복
cvscores = []
Fold = 1
results = np.zeros((20480,10))

def lr_decay(epoch):    #lrv >> learning rate 비율 점차 감소
    return initial_learningrate * 0.99 ** epoch



for train, val in kf.split(A_x): # 50번 반복
    # if Fold<25:
    #   Fold+=1
    #   continue
    initial_learningrate=2e-3  
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=50)    
    lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=25, verbose=1, mode='max')  
    filepath_val_acc="../data/DACON_vision2/cp/vision1_a_"+str(Fold)+".ckpt"    # .ckpt : 모델체크포인트만 저장하는 확장자
    checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)

    gc.collect()
    bek.clear_session()
    print ('Fold: ',Fold)
    
    X_train = x_train224[train]
    X_val = x_train224[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    Y_train = A_y[train]
    Y_val = A_y[val]

    model = create_model()

    batch = 2

    training_generator = datagen.flow(X_train, Y_train, batch_size=batch,seed=7,shuffle=True)   # sedd : random_state와 유사한 기능
    validation_generator = valgen.flow(X_val, Y_val, batch_size=batch,seed=7,shuffle=True)
    model.fit(training_generator,epochs=150,callbacks=[lr,es,checkpoint_val_acc],
               shuffle=True,
               validation_data=validation_generator,
               steps_per_epoch =len(X_train)//batch     # steps_per_epoch : 한 epoch에 사용한 스텝 수. 훈련샘플수 % 배치사이즈
               )

    del X_train
    del X_val
    del Y_train
    del Y_val

    gc.collect()
    bek.clear_session()
'''
    model.load_weights(filepath_val_acc)
    results = results + model.predict(test_224)
    
    Fold = Fold +1

submission['digit'] = np.argmax(results, axis=1)
# model.predict(x_test)
submission.head()
submission.to_csv('../data/DACON_vision2/0225_1_sub.csv', index=False)
'''