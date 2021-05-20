import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import librosa
import warnings
import datetime 

start_now = datetime.datetime.now()
warnings.filterwarnings("ignore")

sample_submission = pd.read_csv("E:\\data\\classification_voice\\sample_submission.csv")
print(sample_submission.iloc[:,1:].shape)  # (6100, 6)

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

"""
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

"""
"""
# npy파일로 저장된 데이터를 불러옵니다.
africa_train_data = np.load("E:\\data\\classification_voice\\npy\\africa_npy.npy", allow_pickle = True)
australia_train_data = np.load("E:\\data\\classification_voice\\npy\\australia_npy.npy", allow_pickle = True)
canada_train_data = np.load("E:\\data\\classification_voice\\npy\\canada_npy.npy", allow_pickle = True)
england_train_data = np.load("E:\\data\\classification_voice\\npy\\england_npy.npy", allow_pickle = True)
hongkong_train_data = np.load("E:\\data\\classification_voice\\npy\\hongkong_npy.npy", allow_pickle = True)
us_train_data = np.load("E:\\data\\classification_voice\\npy\\us_npy.npy", allow_pickle = True)

test_data = np.load("E:\\data\\classification_voice\\npy\\test_npy.npy", allow_pickle = True)

train_data_list = [africa_train_data, australia_train_data, canada_train_data, england_train_data, hongkong_train_data, us_train_data]
for i in range(6) :
    print(len(train_data_list[i]))
# 2500
# 1000
# 1000
# 10000
# 1020
# 10000


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
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def block(input_, units = 32, dropout_rate = 0.5):
    
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(input_)
    x = BatchNormalization()(x)
    x_res = x
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)
    
    return x

def second_block(input_, units = 64, dropout_rate = 0.5):
    
    x = Convolution2D(units, 1, padding ="same", activation = "relu")(input_)
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = Convolution2D(units * 4, 1, padding ="same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Convolution2D(units, 1, padding ="same", activation = "relu")(x)
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = Convolution2D(units * 4, 1, padding ="same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Convolution2D(units, 1, padding = "same", activation = "relu")(x)
    x = Convolution2D(units, 3, padding ="same", activation = "relu")(x)
    x = Convolution2D(units * 4, 1, padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    x = AveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x)
    
    return x

def build_fn():
    dropout_rate = 0.3
    
    in_ = Input(shape = (train_x.shape[1:]))
    
    block_01 = block(in_, units = 32, dropout_rate = dropout_rate)
    block_02 = block(block_01, units = 64, dropout_rate = dropout_rate)
    block_03 = block(block_02, units = 128, dropout_rate = dropout_rate)

    block_04 = second_block(block_03, units = 64, dropout_rate = dropout_rate)
    block_05 = second_block(block_04, units = 128, dropout_rate = dropout_rate)

    x = Flatten()(block_05)

    x = Dense(units = 128, activation = "relu")(x)
    x = BatchNormalization()(x)
    x_res = x
    x = Dropout(rate = dropout_rate)(x)

    x = Dense(units = 128, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Add()([x_res, x])
    x = Dropout(rate = dropout_rate)(x)

    model_out = Dense(units = 6, activation = 'softmax')(x)
    model = Model(in_, model_out)
    return model

split = StratifiedKFold(n_splits = 8, shuffle = True, random_state = 42)

pred = []
pred_ = []

def cov_type(data):
    return np.int(data)

time = 1
for train_idx, val_idx in split.split(train_x, train_y):
    x_train, y_train = train_x[train_idx], train_y[train_idx]
    x_val, y_val = train_x[val_idx], train_y[val_idx]

    model = build_fn()
    model.compile(optimizer = keras.optimizers.Adam(0.002),
                 loss = keras.losses.SparseCategoricalCrossentropy(),
                 metrics = ['acc'])

    history = model.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), epochs = 1)
    print("*******************************************************************")
    # pred.append(model.predict(test_x))
    pred_.append(np.argmax(model.predict(test_x), axis = 0))
    pred = model.predict(test_x)

    print(pred.shape)   # (6100, 6)
    print(pred)        
    # [[8.54439131e-05 9.43848645e-05 1.25233745e-02 3.74446303e-01
    # 7.01941724e-04 6.12148523e-01]
    # [1.29791155e-01 1.92947742e-02 1.25951739e-02 4.20782626e-01
    # 2.35983524e-02 3.93937945e-01]
    # [1.01489954e-01 5.18918075e-02 3.31521034e-02 3.73656929e-01
    # 1.86942890e-02 4.21114892e-01]
    # ...
    # [1.53635606e-01 5.03998436e-02 4.62527499e-02 3.09181273e-01
    # 1.88401248e-02 4.21690404e-01]
    # [1.48892432e-01 7.52632171e-02 6.26772940e-02 3.29950958e-01
    # 1.40392007e-02 3.69176924e-01]
    # [3.01615924e-01 5.77564985e-02 4.10375558e-02 3.52895677e-01
    # 1.37959430e-02 2.32898489e-01]]

    sample_submission.iloc[:,1:] = pred
    sample_submission.to_csv(f"E:\\data\\classification_voice\\baseline_0520_{time}.csv", index=False)

    print(sample_submission.head())
    time += 1
    print("*******************************************************************")
"""
# csv 합쳐서 평균내서 가장 높은 거에만 1로 적기
kf1 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_1.csv")
kf2 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_2.csv")
kf3 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_3.csv")
kf4 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_4.csv")
kf5 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_5.csv")
kf6 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_6.csv")
kf7 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_7.csv")
kf8 = pd.read_csv("E:\\data\\classification_voice\\baseline_0520_8.csv")

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
    sample_submission.to_csv(f"E:\\data\\classification_voice\\baseline_0520_mean.csv", index=False)
    print("=============")


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time) 


# 집컴
# baseline_0520_mean.csv
# 1.3098405569