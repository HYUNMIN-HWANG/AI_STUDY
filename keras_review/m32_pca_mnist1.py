import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000, 28*28)
print(x.shape)

y = np.append(y_train, y_test, axis=0)
print(y.shape)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95)+1
print("cumsum >= 0.95", cumsum > 0.95)
print("d : ", d)

pca = PCA(n_components=154)
x2 = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=47)

from sklearn.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. Modeling
from sklearn.kears.models import Sequential
from skelearn.kears.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, input_shape=(x_train.shape[1],), activaion='relu'))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. Train

#4. Score, Predict
