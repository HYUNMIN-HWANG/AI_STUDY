import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import KFold

# kfold 

######################################################
# File Load
dataset = pd.read_csv('../data/DACON_vision2/dirty_mnist_2nd_answer.csv')
y_df =  dataset.iloc[:,:]
print(y_df.shape)  # (50000, 27)

sub = pd.read_csv('../data/DACON_vision2/sample_submission.csv')
print(sub.shape)    # (5000, 27)

######################################################


#1. DATA
x = np.load('../data/DACON_vision2/npy/vision_x6.npy')
y = np.load('../data/DACON_vision2/npy/vision_y6.npy')
x_pred = np.load('../data/DACON_vision2/npy/vision_x_pred6.npy')
print("<==complete load==>")

print(x.shape, y.shape, x_pred.shape) # (50000, 80, 80, 1) (50000, 26) (5000, 80, 80, 1)

kf_times = 5
kf = KFold(n_splits=kf_times, shuffle=True, random_state=1)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    rotation_range=5,
    zoom_range=0.2,
    fill_mode='nearest'
)
etc_datagen = ImageDataGenerator(rescale=1./255)

#2. Modeling
def mymodel () :
    model = Sequential()
    model.add(Conv2D(64, (2,2), padding='same', input_shape=(x_train.shape[1],x_train.shape[2], x_train.shape[3]), activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(AveragePooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization()) 
    model.add(AveragePooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization()) 
    model.add(Dense(1, activation='sigmoid'))

    return model

# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# alphabet = ['c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# alphabet = ['j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
alphabet = ['q','r','s','t','u','v','w','x','y','z']

val_loss_list = []
val_acc_list = []



for alph in alphabet :  # 한 알파벳씩 검증
    print('<<<<<<<<<<<< ', alph, ' predict start >>>>>>>>>>>>')
    
    val_loss_least = 100

    y = y_df.loc[:,alph]

    y_pred_list = []
    y_result = 0

    kf_n = 1
    for train_index, test_index in kf.split(x, y) :
        print("\n** Kfold %d 번째 실행 중 **" % kf_n)
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9, shuffle=True, random_state=47)

        batch = 16

        train_generator = train_datagen.flow(x_train, y_train, batch_size=batch, seed=42)
        test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch, seed=42)
        valid_generator = etc_datagen.flow(x_valid, y_valid)
        pred_generator = etc_datagen.flow(x_pred)

        print(x_train.shape, x_test.shape, x_valid.shape) # (36000, 80, 80, 1) (10000, 80, 80, 1) (4000, 80, 80, 1)
        print(y_train.shape, y_test.shape, y_valid.shape) # (36000,) (10000,) (4000,)
    
        #2. Modeling
        model = mymodel()

        #3. Compile, Train

        path = f'../data/DACON_vision2/cp/vision_0222_1_{alph}.hdf5'
        cp = ModelCheckpoint(filepath=path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
        lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, mode='min')

        model.compile(
            loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc']
        )

        hist = model.fit_generator(
            train_generator, epochs=50,
            steps_per_epoch = len(x_train) // batch ,
            validation_data=valid_generator, callbacks=[es, lr, cp]
        )

        print("min.history[val_loss] : ", min(hist.history['val_loss']))

        if val_loss_least > min(hist.history['val_loss']) :
            # print("val_loss_least save")
            val_loss_least = min(hist.history['val_loss'])
            model.save(f'../data/DACON_vision2/cp/vision_0222_1_{alph}_final.hdf5')
        
        kf_n += 1

    # loss값이 가장 작은 모델을 돌린다.
    model = load_model(f'../data/DACON_vision2/cp/vision_0222_1_{alph}_final.hdf5')

    #4. Evaluate, Predict
    loss, acc = model.evaluate(test_generator)
    print("loss : ", loss)
    print("acc : ", acc)  

    val_loss_list.append(loss)
    val_acc_list.append(acc)

    # loss :  
    # acc :  

    y_pred = model.predict(pred_generator)
    # print(y_pred.shape) # 
    y_result= np.where(y_pred < 0.5, 0, 1)
    sub.loc[:,alph] = y_result  # 예측값을 sub 파일에 저장
    sub.to_csv('../data/DACON_vision2/sub_0222_1.csv', index=False)
    print(sub.head())

    print("*******", alph, " predict end *******\n")

print("mean loss : ", sum(val_loss_list)/26*kf_times)
print("mean acc : ", sum(val_acc_list)/26*kf_times)

# mean loss : 
# mean acc :  

sub.to_csv('../data/DACON_vision2/sub_0222_1.csv', index=False)
print(sub.head())


# sub
# score 	
