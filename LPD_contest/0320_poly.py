import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50, ResNet101, EfficientNetB2
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.resnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA


start_now = datetime.datetime.now()

z = 0
# pca = PCA()
pca = PCA(n_components=161, svd_solver='full')


pca_xtrain_list = []
pca_xvalid_list = []

for i in range(10) :
    print(z,">>>>>>>>>>>>>>>>>")
    
    for j in range(5) :
        poly_xtrain_list = []
        poly_xvalid_list = []
        start = i * 100 + j * 20
        end = start + 20

        ### npy load
        x_data = np.load(f'../data/LPD_competition/npy/data_x_{start}_{end}.npy', allow_pickle=True)
        x_data = np.resize(x_data, (960, 16*16*3))
        print(x_data.shape)    # (960, 1200)

        y_data = np.load(f'../data/LPD_competition/npy/data_y_{start}_{end}.npy', allow_pickle=True)
        y_data = y_data - start
        print(y_data.shape)    # (960,)
        # print(y_data)
        
        x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=42)
        print(x_train.shape, x_valid.shape) # (864, 1200) (96, 1200)
        print(y_train.shape, y_valid.shape) # (864,) (96,)

        x_train = x_train / 255.
        x_valid = x_valid / 255.
        
        # 특성강조
        x_train = np.where((x_train < 160/255.), 0, x_train)
        x_valid = np.where((x_valid < 160/255.), 0, x_valid)

        # polynomial
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_x_train = poly.fit_transform(x_train)
        poly_x_valid = poly.fit_transform(x_valid)
        print(poly_x_train.shape)   #(864, 296064)
        print(poly_x_valid.shape)   #(96, 296064)

        poly_xtrain_list.append(poly_x_train)
        poly_xvalid_list.append(poly_x_valid)

    print("============")
    # 다섯개씩 묶기
    poly_xtrain_list = np.array(poly_xtrain_list)
    poly_xvalid_list = np.array(poly_xvalid_list)
    poly_xtrain_list = np.resize(poly_xtrain_list, (poly_xtrain_list.shape[0]*poly_xtrain_list.shape[1],poly_xtrain_list.shape[2]))
    poly_xvalid_list = np.resize(poly_xvalid_list, (poly_xvalid_list.shape[0]*poly_xvalid_list.shape[1],poly_xvalid_list.shape[2]))

    print(poly_xtrain_list.shape)   # (4320, 296064)
    print(poly_xvalid_list.shape)   # (480, 296064)
    # pca.fit(poly_x_train)
    # cumsum = np.cumsum(pca.explained_variance_ratio_)
    # d = np.argmax(cumsum >= 0.95) + 1
    # print("cumsum >= 0.95", cumsum > 0.95)
    # print("d : ", d)    # 161

    pca_x_train = pca.fit_transform(poly_xtrain_list)
    pca_x_valid = pca.fit_transform(poly_xvalid_list)
    print(pca_x_train.shape)    # (4320, 161)
    print(pca_x_valid.shape)    # (480, 161)

    pca_xtrain_list.append(pca_x_train)
    pca_xvalid_list.append(pca_x_valid)

    z += 1

pca_xtrain_list = np.array(pca_xtrain_list)
pca_xvalid_list = np.array(pca_xvalid_list)
print(pca_xtrain_list.shape)
print(pca_xvalid_list.shape)


'''
    model = XGBClassifier(n_jobs = -1, use_label_encoder=False, learning_rate=0.01, n_estimators=500)

    model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss', eval_set =[(x_train, y_train), (x_valid, y_valid)], early_stopping_rounds=20)

    result = model.score(x_valid, y_valid)
    print("model.score : ", result)

    result_list.append(result)

    # y_pred = model.predict(x_pred)
    # print(y_pred[:40])
    # print(y_pred.shape)

    # submission['prediction'] = y_pred
    # submission.to_csv('../data/LPD_competition/sub_0319_1.csv',index=True)
    z += 1



#4. Predict

print("predict >>>>>>>>>>>>>> ")
submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

model = load_model('../data/LPD_competition/cp/cp_0319_1_b2.hdf5')

result = model.predict(pred_generator, verbose=True)
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))

submission['prediction'] = np.argmax(result, axis = 1)
submission.to_csv('../data/LPD_competition/sub_0319_1.csv',index=True)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >> 

# score 
'''