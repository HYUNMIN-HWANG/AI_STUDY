import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import librosa
import warnings
import datetime 
import tensorflow as tf

'''
시도해봐야 할 것
- cp 저장하는 거, kf 마다 다르게 저장
- sr 22050으로 해보기
- melspectrogram 파라미터 조정
- 각 6개국의 데이터 수 똑같이 맞추기
- 
'''

# gpu failed init~~ 에 관한 에러 해결
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

start_now = datetime.datetime.now()
warnings.filterwarnings("ignore")

sample_submission = pd.read_csv("E:\\data\\classification_voice\\sample_submission.csv")
print(sample_submission.iloc[:,1:].shape)  # (6100, 6)
'''
africa_train_paths = glob("E:\\data\\classification_voice\\train\\africa\\*.wav")
australia_train_paths = glob("E:\\data\\classification_voice\\train\\australia\\*.wav")
canada_train_paths = glob("E:\\data\\classification_voice\\train\\canada\\*.wav")
england_train_paths = glob("E:\\data\\classification_voice\\train\\england\\*.wav")
hongkong_train_paths = glob("E:\\data\\classification_voice\\train\\hongkong\\*.wav")
us_train_paths = glob("E:\\data\\classification_voice\\train\\us\\*.wav")

path_list = [africa_train_paths, australia_train_paths, canada_train_paths,
             england_train_paths, hongkong_train_paths, us_train_paths]

# 파일 개수
for i in range(6) : 
    print(len(path_list[i]))  
# 2500
# 1000
# 1000
# 10000
# 1020
# 10000
'''

# glob로 test data의 path를 불러올때 순서대로 로드되지 않을 경우를 주의해야 합니다.
# test_ 데이터 프레임을 만들어서 나중에 sample_submission과 id를 기준으로 merge시킬 준비를 합니다.

def get_id(data):
    return np.int(data.split("\\")[4].split(".")[0])

test_ = pd.DataFrame(index = range(0, 6100), columns = ["path", "id"])
test_["path"] = glob("E:\\data\\classification_voice\\test\\*.wav")
test_["id"] = test_["path"].apply(lambda x : get_id(x))

print(test_.head())
#                                          path    id
# 0     E:\data\classification_voice\test\1.wav     1
# 1    E:\data\classification_voice\test\10.wav    10
# 2   E:\data\classification_voice\test\100.wav   100
# 3  E:\data\classification_voice\test\1000.wav  1000
# 4  E:\data\classification_voice\test\1001.wav  1001


def load_data(paths):

    result = []
    for path in tqdm(paths):
        # sr = 16000이 의미하는 것은 1초당 16000개의 데이터를 샘플링 한다는 것입니다.
        data, sr = librosa.load(path, sr = 16000)
        result.append(data)
    result = np.array(result) 
    # 메모리가 부족할 때는 데이터 타입을 변경해 주세요 ex) np.array(data, dtype = np.float32)

    return result

# train 데이터를 로드하기 위해서는 많은 시간이 소모 됩니다.
# 따라서 추출된 정보를 npy파일로 저장하여 필요 할 때마다 불러올 수 있게 준비합니다.

# os.mkdir("./npy_data")

# africa_train_data = load_data(africa_train_paths)
# np.save("E:\\data\\classification_voice\\npy\\africa_npy", africa_train_data)

# australia_train_data = load_data(australia_train_paths)
# np.save("E:\\data\\classification_voice\\npy\\australia_npy", australia_train_data)

# canada_train_data = load_data(canada_train_paths)
# np.save("E:\\data\\classification_voice\\npy\\canada_npy", canada_train_data)

# england_train_data = load_data(england_train_paths)
# np.save("E:\\data\\classification_voice\\npy\\england_npy", england_train_data)

# hongkong_train_data = load_data(hongkong_train_paths)
# np.save("E:\\data\\classification_voice\\npy\\hongkong_npy", hongkong_train_data)

# us_train_data = load_data(us_train_paths)
# np.save("E:\\data\\classification_voice\\npy\\us_npy", us_train_data)

# test_data = load_data(test_["path"])
# np.save("E:\\data\\classification_voice\\npy\\test_npy", test_data)


# npy파일로 저장된 데이터를 불러옵니다.
africa_train_data = np.load("E:\\data\\classification_voice\\npy\\africa_npy.npy", allow_pickle = True)
australia_train_data = np.load("E:\\data\\classification_voice\\npy\\australia_npy.npy", allow_pickle = True)
canada_train_data = np.load("E:\\data\\classification_voice\\npy\\canada_npy.npy", allow_pickle = True)
england_train_data = np.load("E:\\data\\classification_voice\\npy\\england_npy.npy", allow_pickle = True)
hongkong_train_data = np.load("E:\\data\\classification_voice\\npy\\hongkong_npy.npy", allow_pickle = True)
us_train_data = np.load("E:\\data\\classification_voice\\npy\\us_npy.npy", allow_pickle = True)

test_data = np.load("E:\\data\\classification_voice\\npy\\test_npy.npy", allow_pickle = True)

train_data_list = [africa_train_data, australia_train_data, canada_train_data, england_train_data, hongkong_train_data, us_train_data]
# train_data_list = [africa_train_data[:500], australia_train_data[:500], canada_train_data[:500], england_train_data[:500], hongkong_train_data[:500], us_train_data[:500]]
for i in range(6) :
    print(len(train_data_list[i]))
# 2500
# 1000
# 1000
# 10000
# 1020
# 10000

# 500
# 500
# 500
# 500
# 500
# 500

# 이번 대회에서 음성은 각각 다른 길이를 갖고 있습니다.
# baseline 코드에서는 음성 중 길이가 가장 작은 길이의 데이터를 기준으로 데이터를 잘라서 사용합니다.

def get_mini(data):

    mini = 9999999
    for i in data:
        if len(i) < mini:
            mini = len(i)

    return mini

#음성들의 길이를 맞춰줍니다.

def set_length(data, d_mini):

    result = []
    for i in data:
        result.append(i[:d_mini])
    result = np.array(result)

    return result

#feature를 생성합니다.

def get_feature(data, sr = 16000, n_fft = 256, win_length = 200, hop_length = 160, n_mels = 64):
    mel = []
    for i in data:
        # win_length 는 음성을 작은 조각으로 자를때 작은 조각의 크기입니다.
        # hop_length 는 음성을 작은 조각으로 자를때 자르는 간격을 의미합니다.
        # n_mels 는 적용할 mel filter의 개수입니다.
        mel_ = librosa.feature.melspectrogram(i, sr = sr, n_fft = n_fft, win_length = win_length, hop_length = hop_length, n_mels = n_mels)
        mel.append(mel_)
    mel = np.array(mel)
    mel = librosa.power_to_db(mel, ref = np.max)

    mel_mean = mel.mean()
    mel_std = mel.std()
    mel = (mel - mel_mean) / mel_std

    return mel

train_x = np.concatenate(train_data_list, axis= 0)
test_x = np.array(test_data)

# 음성의 길이 중 가장 작은 길이를 구합니다.

train_mini = get_mini(train_x)
test_mini = get_mini(test_x)

mini = np.min([train_mini, test_mini])

# data의 길이를 가장 작은 길이에 맞춰 잘라줍니다.

train_x = set_length(train_x, mini)
test_x = set_length(test_x, mini)

# librosa를 이용해 feature를 추출합니다.

train_x = get_feature(data = train_x)
test_x = get_feature(data = test_x)

train_x = train_x.reshape(-1, train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(-1, test_x.shape[1], test_x.shape[2], 1)

# train_data의 label을 생성해 줍니다.

train_y = np.concatenate((np.zeros(len(africa_train_data), dtype = np.int),
                         np.ones(len(australia_train_data), dtype = np.int),
                         np.ones(len(canada_train_data), dtype = np.int) * 2,
                         np.ones(len(england_train_data), dtype = np.int) * 3,
                         np.ones(len(hongkong_train_data), dtype = np.int) * 4,
                         np.ones(len(us_train_data), dtype = np.int) * 5), axis = 0)

print(train_x.shape, train_y.shape, test_x.shape)   # (25520, 64, 501, 1) (25520,) (6100, 64, 501, 1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Convolution2D, BatchNormalization, Flatten,
                                     Dropout, Dense, AveragePooling2D, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def build_fn():
    model = MobileNet(
    include_top=True,
    input_shape=(64, 501,1),
    classes=6,
    pooling=None,
    weights=None,
    )
    return model

def cov_type(data):
    return np.int(data)
    
split = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 42)

cp_path = 'E:\\data\\classification_voice\\cp\\0521_mobilnet1.h5'
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1, mode='min')
cp = ModelCheckpoint(cp_path, monitor='val_loss', verbose=1, save_best_only=True, mode=True)

time = 1
batch = 8
for train_idx, val_idx in split.split(train_x, train_y):
    x_train, y_train = train_x[train_idx], train_y[train_idx]
    x_val, y_val = train_x[val_idx], train_y[val_idx]

    model = build_fn()
    model.summary()

    model.compile(optimizer = keras.optimizers.Adam(0.002),
                 loss = keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ['acc'])

    history = model.fit(x = x_train, y = y_train, batch_size=batch, validation_data = (x_val, y_val), epochs = 60, callbacks=[es, rl, cp])
    result = model.evaluate(x_val, y_val, batch_size=batch)
    print("loss ", result[0])
    print("acc ", result[1])
    print("*******************************************************************")
   
    pred = model.predict(test_x)

    sample_submission.iloc[:,1:] = pred
    sample_submission.to_csv(f"E:\\data\\classification_voice\\mobilenet_0521_{time}.csv", index=False)

    print(sample_submission.head())
    time += 1
    print("*******************************************************************")

# csv 합쳐서 평균내서 가장 높은 거에만 1로 적기
kf1 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_1.csv")
kf2 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_2.csv")
kf3 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_3.csv")
kf4 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_4.csv")
kf5 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_5.csv")
kf6 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_6.csv")
kf7 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_7.csv")
kf8 = pd.read_csv("E:\\data\\classification_voice\\mobilenet_0521_8.csv")

# 각 컬럼마다 8개 파일의 평균값을 넣는다.
for col in range(1,7) :
    col_mean = []
    col_mean.append(kf1.iloc[:,col])
    col_mean.append(kf2.iloc[:,col])
    col_mean.append(kf3.iloc[:,col])
    col_mean.append(kf4.iloc[:,col])
    col_mean.append(kf5.iloc[:,col])
    col_mean.append(kf6.iloc[:,col])
    col_mean.append(kf7.iloc[:,col])
    col_mean.append(kf8.iloc[:,col])
    print(col_mean)
    mean = np.mean(col_mean, axis=0)
    print(mean)
    sample_submission.iloc[:,col] = mean
    sample_submission.to_csv(f"E:\\data\\classification_voice\\mobilenet_0521_mean1.csv", index=False)
    print("=============")


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 

# time >>  5:02:08.761267
# 집컴
# mobilenet_0521_mean1.csv
# score	2.9818777319	