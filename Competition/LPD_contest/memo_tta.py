import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D, Input, GaussianDropout
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

#0. 변수
filenum = 18
img_size = 192
batch = 16
seed = 42
epochs = 1000
train_dir = '../data/lpd/train_new2'
test_dir = '../data/lpd/test_new'
model_path = '../data/model/lpd_{0:03}.hdf5'.format(filenum)
save_folder = '../data/lpd/submit_{0:03}'.format(filenum)
sub = pd.read_csv('../data/lpd/sample.csv', header = 0)
es = EarlyStopping(patience = 7)
lr = ReduceLROnPlateau(factor = 0.25, patience = 3, verbose = 1)
cp = ModelCheckpoint(model_path, save_best_only= True)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

#1. 데이터
train_gen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    preprocessing_function= preprocess_input
)

test_gen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
    width_shift_range= 0.05,
    height_shift_range= 0.05
)

# Found 58000 images belonging to 1000 classes.
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size = (img_size, img_size),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'training'
)

# Found 14000 images belonging to 1000 classes.
val_data = train_gen.flow_from_directory(
    train_dir,
    target_size = (img_size, img_size),
    class_mode = 'sparse',
    batch_size = batch,
    seed = seed,
    subset = 'validation'
)

# Found 72000 images belonging to 1 classes.
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size = (img_size, img_size),
    class_mode = None,
    batch_size = batch,
    shuffle = False
)

#2. 모델
eff = EfficientNetB4(include_top = False, input_shape=(img_size, img_size, 3))
eff.trainable = True

a = eff.output
a = Dense(2048, activation= 'swish') (a)
a = Dropout(0.3) (a)
a = GlobalAveragePooling2D() (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = eff.input, outputs = a)

#3. 컴파일 훈련
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])
model.fit(train_data, steps_per_epoch = len(train_data), validation_data= val_data, validation_steps= len(val_data),\
    epochs = epochs, callbacks = [es, cp, lr])

model = load_model(model_path)

#4. 평가 예측
'''
result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - mode')
    pred = model.predict(test_data, steps = len(test_data))
    pred = np.argmax(pred, 1)
    result.append(pred)

    print(f'{tta+1} 번째 제출 파일 저장하는 중')
    temp = np.array(result)
    temp = np.transpose(result)

    temp_mode = stats.mode(temp, axis = 1).mode
    sub.loc[:, 'prediction'] = temp_mode
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)

    temp_count = stats.mode(temp, axis = 1).count
    for i, count in enumerate(temp_count):
        if count < tta/2.:
            print(f'{tta+1} 반복 중 {i} 번째는 횟수가 {count} 로 {(tta+1)/2.} 미만!')
'''
cumsum = np.zeros([72000, 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps = len(test_data)) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    temp_percent = np.max(temp, 1)

    count = 0
    i = 0
    for percent in temp_percent:
        if percent < 0.3:
            print(f'{i} 번째 테스트 이미지는 {percent}% 의 정확도를 가짐')
            count += 1
        i += 1
    print(f'TTA {tta+1} : {count} 개가 불확실!')
    count_result.append(count)
    print(f'기록 : {count_result}')
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv(save_folder + '/sample_{0:03}_{1:02}.csv'.format(filenum, (tta+1)), index = False)