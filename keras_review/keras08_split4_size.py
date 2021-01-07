from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

x = np.array(range(1,11)) #1부터 10까지
y = np.array(range(1,11)) #1부터 10까지

from sklearn.model_selection import train_test_split
#1
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
#2
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=True)
#3
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.1)
#4
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.3)
