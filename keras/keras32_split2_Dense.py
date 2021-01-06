import numpy  as np

a = np.array(range(1, 11))
size = 5

# Dense 모델을 구성하시오

def split_x(seq, size) :
    aaa = []  
    for i in range(len(seq) - size + 1) :       # range(len(seq) - size + 1) : 반복횟수(= 행의 개수), # size : 열의 개수
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)  # (6, 5)

'''
dataset
        X     |  Y
======================
[[ 1  2  3  4 | 5]
 [ 2  3  4  5 | 6]
 [ 3  4  5  6 | 7]
 [ 4  5  6  7 | 8]
 [ 5  6  7  8 | 9]
 [ 6  7  8  9 | 10]]
'''

#1. DATA

x = dataset[:,:4] # [0:6,0:4]
# print(x) 
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
y = dataset[:,-1:] # [0:6,4:], [:, -1:]
# print(y)
# [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]

# x_pred = np.array([7,8,9,10])
x_pred = dataset[-1:,1:]

print(x.shape)  # (6, 4)  
print(y.shape)  # (6, 1)

# x = x.reshape(6, 4, 1)
# x_pred = x_pred.reshape(1, 4, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=200, batch_size=1)

#4. Evaluate, Predcit
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)


y_pred = model.predict(x_pred)
print("예측값 : ", y_pred)

# LSTM
# loss :  0.00011622656165855005
# mae :  0.009383519180119038
# 예측값 :  [[11.047476]]

# Dense
# loss :  0.0010288774501532316
# mae :  0.028726181015372276
# 예측값 :  [[11.134745]]
