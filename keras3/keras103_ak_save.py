# autokeras 
# pip install autokeras
# model = ak.ImageClassifier

import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,    # epochs만큼을 2번 반복했다.
    loss='mse',
    # metrcis=['mse']   # 회귀모델일 때
    metrics=['acc']
)
# 특징 1 : input shape 지정을 안 했는데도 돌아간다.
# 특징 2 : y 데이터에 onehotencoding 안했는데도 돌아감
# 특징 3 : y 데이터에 onehotencoding 한 후도 잘 돌아감 < onehotencoding을 해도 되고 안해도 된다.
# 특징 4 : 체크포인트가 자동으로 생성된다.
# 특징 5 : validation split, callbacks 기능을 사용할 수 있다.

# model.summary()   
# # ImageClassifier - summary 기능 사용할 수 없다. 왜? 아직 모델이 완성된 모델이 아니기 때문에  


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=6, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, mode='min')
path = '../data/cp/'
cp = ModelCheckpoint(filepath = path, monitor='val_loss', save_weights_only=True, save_best_only=True)

model.fit(x_train, y_train, epochs=1, validation_split=0.2, 
          callbacks=[es, lr, cp])

results = model.evaluate(x_test, y_test)

print(results)  # [0.07784847915172577, 0.973800003528595]

# model.summary()
# 모델이 확정 된 다음에는 summary를 사용할 수 있을까? Nope, 그래도 summary 없다.

model2 = model.export_model()
# export_model : 원래의 모델 형식으로 내보내준다.
model2.save('../data/autokeras/save_model.h5')   
# AttributeError: 'ImageClassifier' object has no attribute 'save'
# -> ImageClassifier 그대로는 모델 저장 못한다.
# 위에서 model.export_model() 를 넣어야 함