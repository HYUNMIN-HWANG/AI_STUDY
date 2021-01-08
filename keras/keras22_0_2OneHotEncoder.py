import numpy as np
x = np.array([3,6,5])
print(x.shape)  # (3,)

# (1)
from tensorflow.keras.utils import to_categorical

x = to_categorical(x) 
print(x)
'''
  0  1  2  3  4  5  6
[[0. 0. 0. 1. 0. 0. 0.]  --> 3
 [0. 0. 0. 0. 0. 0. 1.]  --> 6
 [0. 0. 0. 0. 0. 1. 0.]] --> 5
 '''
print(x.shape)    # (3, 7)

# (2)
y = np.array([3,6,5])
from sklearn.preprocessing import OneHotEncoder

y = y.reshape(-1,1)
print(y)
'''
[[3]
 [6]
 [5]]
 '''
encoder = OneHotEncoder()
encoder.fit(y)
y = encoder.transform(y).toarray()  #toarray() : list 를 array로 바꿔준다.
print(y)
'''
  3  5  6
[[1. 0. 0.]  --> 3
 [0. 0. 1.]  --> 6
 [0. 1. 0.]] --> 5
 '''
print(y.shape)  # (3, 3)