import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

########### npy save

# train_datagen = ImageDataGenerator(
#     # rescale=1./255,
#     width_shift_range=(-1,1),
#     height_shift_range=(-1,1),
#     rotation_range=5,
#     zoom_range=5,
#     shear_range=0.5,
#     fill_mode='nearest'
# )


test_image = glob('../data/LPD_competition/test/*.jpg')
print(len(test_image))      # 72000


resize_data = []

# for img in test_image :
#     now_img = os.path.basename(img)
#     print("\n", now_img)

#     copy_img = img 
#     img1 = load_img(copy_img , color_mode='rgb', target_size=(128,128)) 
#     img1 = img1.resize((128, 128))
#     img1 = np.array(img1)
#     img1 = img1.reshape(-1, 128, 128,3)
#     # print(img1.shape)   # (1, 128, 128, 3)
#     resize_data.append(img1)


img1=[]
for i in range(0,72000):
    filepath='../lotte/test/%d.jpg'%i
    image2 = Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((image_w, image_h))
    image_data2=asarray(image2)
    # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
    img1.append(image_data2)    

np.save("../data/lotte/npy/test.npy", arr=img1)

x_pred = np.load('../data/lotte/npy/test.npy',allow_pickle=True)

test_datagen = ImageDataGenerator()
xy_test = test_datagen.flow(
    resize_data,
    batch_size=80000,
)   

np.save('../data/LPD_competition/npy/data_x1_test.npy', arr=xy_test)

# MemoryError: Unable to allocate 192. KiB for an array with shape (128, 128, 3) and data type float32