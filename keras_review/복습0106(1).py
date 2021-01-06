# LSTM

#1. DATA
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])  #(13, 3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])    #(13, )

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True,random_state=44)

from sklearn.preprocessing import 

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()


#3. Compile, Train

#4. Evaluate, Predict

