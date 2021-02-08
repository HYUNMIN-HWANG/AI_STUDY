import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


#1. DATA / Preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
# x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x_train = x_train.reshape(60000, 28*28).astype('float32')
x_test = x_test.reshape(10000, 28*28).astype('float32')

#2. Modeling

path = '../data/modelcheckpoint/kk61_4_{val_loss:.4f}'
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verose=1)

#2. Modeling
def build_model(drop=0.5, optimizer='adam', ) :
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def hyperparameter() :
    batches = [8, 32]
    optimizers = ['rmsprop', 'adam']
    dropout = [0.2, 0.3]
    return {"model__batch_size" : batches, "model__optimizer" : optimizers, "model__drop" : dropout}

# model = build_model()
model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=5, validation_split=0.2)
pipe = Pipeline([('scaler',MinMaxScaler()), ('model',model2)])

hyperparameter = hyperparameter()
kf = KFold(n_splits=2, random_state=47)

cv = RandomizedSearchCV(pipe, hyperparameter, cv=kf)

cv.fit(x_train, y_train)

# model save
# cv.best_estimator_.model.save('../data/h5/kk64_model_save.h5')

print("best_params : ", cv.best_params_)
print("best_score : ", cv.best_score_)

acc = cv.score(x_test, y_test)
print("score : ", acc)

# DNN
# best_params :  {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 16, 'activation': 'elu'}
# best_score :  0.7401833236217499
# score :  0.82669997215271

#CNN
# best_params :  {'optimizer': 'rmsprop', 'node': 32, 'kernel_size': 3, 'drop': 0.5, 'batch_size': 32, 'activation': 'relu'}
# best_score :  0.9432166814804077
# score :  0.9645000100135803

# cv_results
csv = cv.cv_results_
print("cv_result : ", cv)
# cv_result :  RandomizedSearchCV(cv=KFold(n_splits=2, random_state=47, shuffle=False),
#                    estimator=<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000017C5A548070>,
#                    param_distributions={'activation': ['relu', 'elu'],
#                                         'batch_size': [16, 32],
#                                         'drop': [0.3, 0.5],
#                                         'kernel_size': [2, 3], 'node': [32, 16],
#                                         'optimizer': ['adam', 'rmsprop']})

