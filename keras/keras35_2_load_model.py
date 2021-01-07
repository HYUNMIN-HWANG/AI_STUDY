# 저장한 모델 불러오기

from tensorflow.keras.models import load_model
model = load_model ('./model/save_keras35.h5')

model.summary()

# 아래 경고 메세지 : 나중에 가중치 저장할 때 확인할 예정
# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
