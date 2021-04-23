# warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.optimizers import RMSprop, Adam
# from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
# from keras.utils import np_utils
import cv2
import gc
from keras import backend as bek
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


########################################
train = pd.read_csv('../data/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

submission = pd.read_csv('../data/DACON_vision1/submission.csv')
print(submission.shape) # (20480, 2)

test = pd.read_csv('../data/DACON_vision1/test.csv')
print(test.shape)   # (20480, 786)
########################################

# vision1에 나온 알파벳들을 훈련시켜보자

# x
def x_preprocess (x) :
    x_train = np.where((x<=20)&(x!=0) ,0.,x)  # 특징이 낮은 것들은 모두 0으로 반환
    x_train = np.where((x>=80) ,252.,x)  # 알파벳만 강조

    x_train = x_train.astype('float32')

    x_train224=np.zeros([2048,56,56,3],dtype=np.float32) # [2048,56,56,3] 의 검은색 배경을 만들어 준다.

    for i, s in enumerate(x_train):
        print(i)
        converted = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
        # cv2.imshow("image", converted)
        # cv2.waitKey(0)
        resized = cv2.resize(converted,(56,56),interpolation = cv2.INTER_CUBIC) # (56, 56)으로 리사이즈
        blur =  cv2.GaussianBlur(resized, (5,5), cv2.BORDER_DEFAULT)
        threshold, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY) 
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        del converted
        x_train224[i] = thresh
        # cv2.imshow("resized_224", x_train224[i])
        # cv2.waitKey(0)
        del thresh
        bek.clear_session()
        gc.collect()

    x_train224 = x_train224/255.
    print(x_train224.shape)    # (2048, 56, 56, 3)
    return x_train224

x_train = train.drop(['id', 'digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1,28,28, 1)
'''
x_train224 = x_preprocess (x_train)
print(x_train224.shape)    # (2048, 56, 56, 3)

# np.save('../data/DACON_vision2/npy/vision1_image.npy', arr=x_train224)
'''
x_train224 = np.load('../data/DACON_vision2/npy/vision1_image.npy')

idg = ImageDataGenerator(height_shift_range=(-1,1), 
                        width_shift_range=(-1,1),
                        rotation_range=120,
                        fill_mode='nearest'
                        )
idg2 = ImageDataGenerator()


# y
y = train['letter']
# y_train = np.zeros((len(y), 1))  # 총 행의수 26
y_letter = np.zeros((len(y),), dtype=int)  # 총 행의수 26

i = 0
for letter in y :
    # print(i, letter)
    # 문자열을 아스키 코드로 변환, A = 0 부터 시작하도록 지정
    asc = int( ord(letter) - 65 )
    y_letter[i] = asc
    # print(y_letter[i])
    i += 1
# print(y_train.shape)    # (2048, )
# print(y_train[:6])      # [11  1 11  3  0  2]

# print(y_train[:20])

x_train, x_test, y_train, y_test = train_test_split(x_train224, y_letter, train_size=0.9, shuffle=True, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=42)

print(x_train.shape, x_test.shape, x_valid.shape)
# (1474, 56, 56, 3) (410, 56, 56, 3) (164, 56, 56, 3)

train_generator = idg.flow(x_train, y_train, batch_size=16)
test_generator = idg2.flow(x_test, y_test, batch_size=16)
valid_generator = idg2.flow(x_valid, y_valid)


#2 modeling
# def modeling(activation1='relu', activation2='relu', activation3='relu', activation4='relu', activation5='relu', activation6='relu') :
# def modeling(lr=0.01) :
# def modeling(filter1=32, filter2=64, node1=128, node2=64) :
# def modeling(kernel1=2, kernel2=2, kernel3=2, kernel4=2) :
def modeling() :
    model = Sequential()
    model.add(Conv2D(128, 3, activation='selu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization()) 
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(2,2))
    # model.add(Dropout(0.2))

    model.add(Conv2D(32, 3, activation='selu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(32, 2, activation='elu', padding='same'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(2,2))
    # model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256, activation='selu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='selu'))
    model.add(BatchNormalization())
    model.add(Dense(26, activation='softmax'))

    # model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.01, epsilon=None), metrics=['acc'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['acc'])
    return model

model = modeling()
# model = KerasClassifier(build_fn=modeling, verbose=1)

# def create_hyperparameters() :   
    # activation1 =['relu','elu','selu']   
    # activation2 =['relu','elu','selu']  
    # activation3 =['relu','elu','selu']  
    # activation4 =['relu','elu','selu']  
    # activation5 =['relu','elu','selu']  
    # activation6 =['relu','elu','selu']  
    # return {"activation1" : activation1, "activation2" : activation2,\
    #     "activation3" : activation3, "activation4" : activation4,\
    #     "activation5" : activation5, "activation6" : activation6}
    # 
    # optimizers = ['rmsprop', 'adam', 'adadelta', 'sgd', 'adadelta','nadam']
    # return {"optimizer" : optimizers}

    # lr = [0.1, 0.01, 0.001, 0.3, 0.03, 0.003]
    # return {"lr" : lr}

    # filter1 = [16, 32, 64, 128, 256]
    # filter2 = [16, 32, 64, 128, 256]
    # node1 = [16, 32, 64, 128, 256]
    # node2 = [16, 32, 64, 128, 256]
    # return {"filter1" : filter1, "filter2" : filter2, "node1" : node1, "node2" : node2}

    # kernel1 = [2,3]
    # kernel2 = [2,3]
    # kernel3 = [2,3]
    # kernel4 = [2,3]
    # return {"kernel1" : kernel1,"kernel2" : kernel2,"kernel3" : kernel3,"kernel4" : kernel4}

#3 Compile, Train

# hyperparameters = create_hyperparameters()
# search = RandomizedSearchCV(model, hyperparameters, cv=4)
# search.fit(x_train, y_train, verbose=1)
# print("best_params : ", search.best_params_)

# best_params :  {'activation6': 'selu', 'activation5': 'selu', 'activation4': 'elu', 'activation3': 'selu', 'activation2': 'relu', 'activation1': 'selu'}
# best_params :  {'optimizer': 'adam'}
# best_params :  {'lr': 0.003}
# best_params :  {'node2': 128, 'node1': 256, 'filter2': 32, 'filter1': 128}
# best_params :  {'kernel4': 2, 'kernel3': 3, 'kernel2': 3, 'kernel1': 3}


path = '../data/DACON_vision2/cp/vision_0225_1_vison1_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, mode='min')
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')

model.fit_generator(train_generator, epochs=1000, validation_data=valid_generator, callbacks=[cp, rl, es])


#4 Evaluate, Predict
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.6132016181945801
# acc :  0.8341463208198547



model = load_model('../data/DACON_vision2/cp/vision_0225_1_vison1_0.3318.hdf5') # <<-- 여기에 데이콘2 테스트 데이터를 넣어서 숫자를 예측한다.

# 테스트 : 40번째 있는 문자를 예측해보자
pred_img = x_train224[49]
# pred_img = cv2.imread('../data/DACON_vision2/contour/1_1.png')
cv2.imshow("pred_img1", pred_img)
cv2.waitKey(0)

pred_img = pred_img.reshape(1, pred_img.shape[0], pred_img.shape[1], pred_img.shape[2])
pred_generator = idg2.flow(pred_img, shuffle=False)

result = model.predict_generator(pred_generator, verbose=True)
print(result.argmax(1))
