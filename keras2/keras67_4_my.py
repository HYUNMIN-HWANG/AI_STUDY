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
# model = load_model('../data/modelcheckpoint/k67_0.664_31.h5')
model = load_model('../data/modelcheckpoint/k67_2_0.653_21.h5')

#3. Compile, Train
# es = EarlyStopping(monitor='val_loss', patience=50, mode='min')
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=30, mode='min')

# model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['acc'])
# model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_valid, y_valid), callbacks=[es, lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss : ", loss)
print("acc : ", acc)

# loss : 
# acc :  

####################[My Image]####################

datagen_2 = ImageDataGenerator(rescale=1./255)

# my image >> x_pred
im = Image.open('../data/image/gender/HHM.jpg')
my = np.asarray(im)
my = np.resize(my, (56, 56, 3))
my = my.reshape(1, 56, 56, 3)
# print(my.shape)     # (1, 56, 56, 3)
x_pred = datagen_2.flow(my)

y_pred = model.predict(x_pred)
y_pred = y_pred[0][0]
# acc_score = accuracy_score(my_answer, y_pred)
# acc_score2 = accuracy_score(my_wrong, y_pred)
# print(y_pred)
# print("여자 0, 남자 1 : ", np.where(y_pred>0.5, 1, 0))
print("남자일 확률은 ",np.round(y_pred*100,2), '%')
print("여자일 확률은 ",np.round((1-y_pred)*100,2), '%')
