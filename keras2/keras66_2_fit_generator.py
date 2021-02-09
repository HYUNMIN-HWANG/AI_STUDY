# ImageDataGenerator 사용법 
# 이미지 전처리
# fit_generator 사용해서 훈련시킨다.

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# [1] 선언
# ImageDataGenerator 선언 >> 이미지를 증폭시킴으로써 더 많은 데이터로 훈련시킬 수 있다.
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 전처리
    horizontal_flip=True,   # 수평
    vertical_flip=True,     # 수직
    width_shift_range=0.1,  # 좌우
    height_shift_range=0.1, # 상하
    rotation_range=5,       # 회전
    zoom_range=1.2,         # 확대
    shear_range=0.7,        # 밀림 정도
    fill_mode='nearest'     # 빈 자리를 유사한 값으로 채운다.
)
test_datagen = ImageDataGenerator(rescale=1./255)   # 전처리만 한다. >> test에서는 이미지 증폭을 할 필요가 없다.

# [2] 데이터화
#   1) flow >> 이미지가 데이터화 되어 있을 때 사용함
#   2) flow_from_directory >> 폴더 안에 있는 파일을 데이터화 한다.
# 폴더 자체를 라벨링할 수 있다. --> (ex) ad : 0, noraml : 1

# train_generator
# ad / normal (앞에 있는 걸 0, 뒤에 있는 걸 1로 라벨링됨)
# x : (80, 150, 150, 1) >> 모두 0부터 1
# y : (80,)             >> ad : 모두 0으로 라벨링 / normal : 모두 1로 라벨링
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',    # 경로 설정 > 해당 폴더에 있는 이미지들을 데이터화 하겠다. > 80개 이미지 전부 0으로 라벨링
    target_size=(150, 150),         # 이미지 크기
    batch_size=5,                  # batch_size 만큼의 이미지가 생성된다. 5장씩 이미지를 자른다.
    class_mode='binary'             # 이진분류 0, 1
)
# 결과 : Found 160 images belonging to 2 classes.

#   [참고]
#   * fit_generator 는 배치사이즈 신경 안써도 된다. 자동으로 연산해준다.
#   * fit 할 때는 배치사이즈 고려해서 잘라줘야 한다.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',    # 경로 설정 > 해당 폴더에 있는 이미지들을 데이터화 하겠다. > 80개 이미지 전부 0으로 라벨링
    target_size=(150, 150),        # 이미지 크기
    batch_size=5,                  # batch_size 만큼의 이미지가 생성된다. 5장씩 이미지를 자른다.
    class_mode='binary'            # 이진분류 0, 1
)
# 결과 : Found 120 images belonging to 2 classes.

#2. Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), activation='relu' , input_shape=(150, 150, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))   # 이진분류

#3. Compile, Train

es = EarlyStopping(monitor='val_loss', patience=40, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.4, mode='min')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# fit_generator 사용해서 훈련시킨다.
history = model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=500, validation_data=xy_test, validation_steps=4, callbacks=[es, lr]
)
# [fit와 차이점]
#   fit_generator : x와 y가 뭉쳐있다.
#   >> xy_train, xy_test : x와 y 통째로 넣어준다.
#   steps_per_epoch : 한 에포 당 돌아가는 횟수 -->> 전체 데이터(160) / 배치 사이즈(5) = steps_per_epoch=32
#   validation_steps : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수

# Graph 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss)
plt.plot(val_loss)
plt.plot(acc)
plt.plot(val_acc)

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['loss','val loss','acc','val acc'])
plt.show()

print("acc : ", acc[-1])            # acc의 가장 마지막 값
print("val_acc : ", val_acc[:-1])   # val_acc 처음부터 끝까지

# acc :  0.7250000238418579
# val_acc :  [0.6000000238418579, 0.550000011920929, 0.44999998807907104, 0.550000011920929, 0.550000011920929, 0.6499999761581421, 0.4000000059604645, 0.44999998807907104, 0.6000000238418579, 0.44999998807907104, 0.44999998807907104, 0.5, 0.949999988079071, 0.6499999761581421, 0.699999988079071, 0.8999999761581421, 1.0, 0.6499999761581421, 0.8999999761581421, 0.8999999761581421, 0.800000011920929, 0.8500000238418579, 0.8999999761581421, 0.949999988079071, 0.699999988079071, 0.8999999761581421, 0.44999998807907104, 0.800000011920929, 0.4000000059604645, 0.6499999761581421, 0.550000011920929, 0.4000000059604645, 0.5, 0.6499999761581421, 0.6000000238418579, 0.5, 0.6000000238418579, 0.6499999761581421, 0.550000011920929, 0.8999999761581421, 0.8999999761581421, 0.800000011920929, 0.8999999761581421]