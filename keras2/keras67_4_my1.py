# pred >> 나를 찍어서 내가 남자인지 여자인지에 대해 >> 이 부분 로직 메일로 제출
# 그때 나오는 acc 명시

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, AveragePooling2D, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import PIL.Image as pilimg
from PIL import Image
from sklearn.metrics import accuracy_score

#1. DATA
# npy load
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
x_valid = np.load('../data/image/gender/npy/keras67_valid_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
y_valid = np.load('../data/image/gender/npy/keras67_valid_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=42)

print(x_train.shape, x_valid.shape, x_test.shape)  # (1319, 56, 56, 3) (347, 56, 56, 3) (70, 56, 56, 3)
print(y_train.shape, y_valid.shape, y_test.shape)  # (1319,) (347,) (70,)

#2. Modeling
# model = load_model('../data/modelcheckpoint/k67_56_0.693.h5')   # 
# model = load_model('../data/modelcheckpoint/k67_56_2_0.691.h5')   # 
# model = load_model('../data/modelcheckpoint/k67_56_3_0.688.h5')   # 
# model = load_model('../data/modelcheckpoint/k67_56_6_0.690.hdf5')   # keep
# model = load_model('../data/modelcheckpoint/k67_0.682_17.h5')   # 
model = load_model('../data/modelcheckpoint/k67_56_7_0.674.hdf5')   # keep



#3. Compile, Train
# es = EarlyStopping(monitor='val_loss', patience=50, mode='min')
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=30, mode='min')

# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
# model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_valid, y_valid), callbacks=[es, lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)

# loss : 
# acc :  

####################[My Image]####################

datagen_2 = ImageDataGenerator(rescale=1./255)

# my image >> x_pred
im1 = Image.open('../data/image/gender/HHM.jpg')   # f
my1 = np.asarray(im1)
my1 = np.resize(my1, (56, 56, 3))
my1 = my1.reshape(1, 56, 56, 3)
my = datagen_2.flow(my1)

im2 = Image.open('../data/image/gender/HL2.jpg')   # f
my2 = np.asarray(im2)
my2 = np.resize(my2, (56, 56, 3))
my2 = my2.reshape(1, 56, 56, 3)
HL = datagen_2.flow(my2)

im3 = Image.open('../data/image/gender/LHL.jpg')   # f
my3 = np.asarray(im3)
my3 = np.resize(my3, (56, 56, 3))
my3 = my3.reshape(1, 56, 56, 3)
LHL = datagen_2.flow(my3)

im4 = Image.open('../data/image/gender/KDW.jpg')   # m
my4 = np.asarray(im4)
my4 = np.resize(my4, (56, 56, 3))
my4 = my4.reshape(1, 56, 56, 3)
KDW = datagen_2.flow(my4)

im5 = Image.open('../data/image/gender/HB.jpg')    # m
my5 = np.asarray(im5)
my5 = np.resize(my5, (56, 56, 3))
my5 = my5.reshape(1, 56, 56, 3)
HB = datagen_2.flow(my5)
######################

my_pred = model.predict(my)
my_pred = my_pred[0][0]
# print(my_pred)
print("당신은   ",np.round((1-my_pred)*100,2), '%의 확률로 여자입니다.')

HL_pred = model.predict(HL)
HL_pred = HL_pred[0][0]
# print(HL_pred)
print("이혜리는 ",np.round((1-HL_pred)*100,2), '%의 확률로 여자입니다.')

LHL_pred = model.predict(LHL)
LHL_pred = LHL_pred[0][0]
# print(LHL_pred)
print("이효리는 ",np.round((1-LHL_pred)*100,2), '%의 확률로 여자입니다.')

KDW_pred = model.predict(KDW)
KDW_pred = KDW_pred[0][0]
# print(KDW_pred)
print("강동원는 ",np.round(KDW_pred*100,2), '%의 확률로 남자입니다.')

HB_pred = model.predict(HB)
HB_pred = HB_pred[0][0]
# print(HB_pred)
print("현빈은   ",np.round(HB_pred*100,2), '%의 확률로 남자입니다.')

# 당신은    50.43 %의 확률로 여자입니다.
# 이혜리는  32.75 %의 확률로 여자입니다.
# 이효리는  39.9 %의 확률로 여자입니다.
# 강동원는  62.47 %의 확률로 남자입니다.
# 현빈은    62.21 %의 확률로 남자입니다.
