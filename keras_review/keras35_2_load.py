# load model

from tensorflow.keras.models import load_model
model = load_model("./mode/save_keras35.h5")
model.add(Dense(10, name='aaa'))
model.add(Dense(1, name='bbb'))
model.summary()