# 인공지능 계의 hello world 라 불리는  mnist

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28)--> 흑백 1 생략 가능 (60000,) 
print(x_test.shape, y_test.shape)   # (10000, 28, 28)                     (10000,)

print(x_train[0])   
print("y_train[0] : " , y_train[0])   # 5
print(x_train[0].shape)               # (28, 28)

plt.imshow(x_train[0], 'gray')  # 0 : black, ~255 : white (가로 세로 색깔)
# plt.imshow(x_train[0]) # 색깔 지정 안해도 나오긴 함
plt.show()  

