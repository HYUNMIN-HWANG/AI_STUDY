from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

#1. DATA
img_dog = load_img('../data/image/vgg/dog3.jpg', target_size=(100,100))
img_cat = load_img('../data/image/vgg/cat3.jpg', target_size=(100,100))
img_lion = load_img('../data/image/vgg/lion3.jpg', target_size=(100,100))
img_suit = load_img('../data/image/vgg/suit3.jpg', target_size=(100,100))

# plt.imshow(img_dog)
# plt.show()

img_dog = img_to_array(img_dog)
img_cat = img_to_array(img_cat)
img_lion = img_to_array(img_lion)
img_suit = img_to_array(img_suit)

print(img_dog.shape)    # (100, 100, 3)

arr_input = np.stack([img_dog, img_cat, img_lion, img_suit])
print(arr_input.shape)  # (4, 100, 100, 3)

#2. Modeling
model = EfficientNetB0()
results = model.predict(arr_input)

print(results)
print(results.shape)

# [[2.0976717e-04 1.3595667e-03 9.4662065e-04 ... 1.4722196e-04
#   7.0132484e-04 1.4100613e-03]
#  [1.5760332e-04 1.0600641e-04 2.8513830e-05 ... 6.9248083e-05
#   1.2905236e-05 7.9719939e-05]
#  [1.6876723e-04 5.5476464e-04 2.3920138e-04 ... 3.8257716e-04
#   2.2996700e-04 8.7386038e-04]
#  [4.4396920e-06 1.2408331e-04 2.6643541e-04 ... 9.3103758e-05
#   2.5766070e-05 2.0713660e-04]]
# (4, 1000)

#3. 이미지 확인
from tensorflow.keras.applications.efficientnet import decode_predictions
decode_results = decode_predictions(results)
print("======================================")
print("0 : ", decode_results[0])
print("1 : ", decode_results[1])
print("2 : ", decode_results[2])
print("3 : ", decode_results[3])
print("======================================")
