from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

print(x_train[0])
print(y_train[:5])
print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)

print("max length : ", max(len(l) for l in x_train))            # 2376
print("mean length : ", sum(map(len, x_train))/len(x_train))    # 145.5398574927633

# plt.hist(y_train, bins=40)
# plt.show()

# preprocessing
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)  # (8982, 46) (2246, 46)

# x
from tensorflow.keras.preprocessing.sequence import pad_sequences
max = 100
x_train = pad_sequences(x_train, maxlen=max, padding='pre')
x_test = pad_sequences(x_test, maxlen=max, padding='pre')

# y

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, LSTM, Conv1D

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=max))
model.add(Conv1D(128, 3))
model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=1, validation_split=0.2)

results = model.evaluate(x_test, y_test)
print("loss : ", results[0])
print("acc : ", results[1])

# loss :  1.9949673414230347
# acc :  0.6602849364280701
