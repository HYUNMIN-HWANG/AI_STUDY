# flow_from_directory : 폴더 구조에 있는 것을 당겨온다.
# save_to_dir : 이미지를 변환한 것을 확인하고 싶다. 

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    batch_size=200,                  # batch_size 만큼의 이미지가 생성된다. 200장씩 이미지를 자른다.
    class_mode='binary'
    , save_to_dir='../data/image/brain_generator/train/'   # 디렉토리에 저장, 이미지 증폭이 랜덤하게 적용된다.
)
# 결과 : Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',    # 경로 설정 > 해당 폴더에 있는 이미지들을 데이터화 하겠다. > 80개 이미지 전부 0으로 라벨링
    target_size=(150, 150),        # 이미지 크기
    batch_size=5,                 # batch_size 만큼의 이미지가 생성된다. 5장씩 이미지를 자른다.
    class_mode='binary'            # 이진분류 0, 1
    , save_to_dir='../data/image/brain_generator/test/'   # 디렉토리에 저장
)
# 결과 : Found 120 images belonging to 2 classes.

# save_to_dir >> 아래 flow_from_directory을 건드릴 때마다 지정된 폴더에 batch_size 개수마다 이미지가 저장된다.
print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][1].shape)
# print(xy_train[1][1])
