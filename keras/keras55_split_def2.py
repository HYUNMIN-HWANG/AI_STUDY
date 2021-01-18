import numpy as np

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],\
                    [11,12,13,14,15,16,17,18,19,20],\
                    [21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)
# 다입력, 다 : 다

def split_xy4(dataset, time_steps, y_column) :
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy4(dataset, 3, 2)
# print(x, "\n", y)
# print(x.shape)      # (7, 3, 2)
# print(y.shape)      # (7, 2)
# y = y.reshape(y.shape[0])   # 벡터 형태로 변환
# print(y.shape)      # (8,)
'''
x : 
[[[ 1 11]
  [ 2 12]
  [ 3 13]]

 [[ 2 12]
  [ 3 13]
  [ 4 14]]

 [[ 3 13]
  [ 4 14]
  [ 5 15]]

 [[ 4 14]
  [ 5 15]
  [ 6 16]]

 [[ 5 15]
  [ 6 16]
  [ 7 17]]

 [[ 6 16]
  [ 7 17]
  [ 8 18]]

 [[ 7 17]
  [ 8 18]
  [ 9 19]]]
y :
 [[23 24]
 [24 25]
 [25 26]
 [26 27]
 [27 28]
 [28 29]
 [29 30]]
'''
#####################################################################################
# 다입력, 다 : 다 (두 번째) - 행으로 자른다.
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],\
                    [11,12,13,14,15,16,17,18,19,20],\
                    [21,22,23,24,25,26,27,28,29,30]])
# print("dataset.shape : ", dataset.shape)     # (3, 10)
dataset = np.transpose(dataset)
# print(dataset)
print("dataset.shape : ", dataset.shape)     # (10, 3)

def split_xy5(dataset, time_steps, y_column) :
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, ]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy5(dataset, 3, 1)
# print(x, "\n", y)
# print(x.shape)      # (7, 3, 3)
# print(y.shape)      # (7, 1, 3)
'''
x : 
[[[ 1 11 21]
  [ 2 12 22]
  [ 3 13 23]]

 [[ 2 12 22]
  [ 3 13 23]
  [ 4 14 24]]

 [[ 3 13 23]
  [ 4 14 24]
  [ 5 15 25]]

 [[ 4 14 24]
  [ 5 15 25]
  [ 6 16 26]]

 [[ 5 15 25]
  [ 6 16 26]
  [ 7 17 27]]

 [[ 6 16 26]
  [ 7 17 27]
  [ 8 18 28]]

 [[ 7 17 27]
  [ 8 18 28]
  [ 9 19 29]]]
y :
 [[[ 4 14 24]]

 [[ 5 15 25]]

 [[ 6 16 26]]

 [[ 7 17 27]]

 [[ 8 18 28]]

 [[ 9 19 29]]

 [[10 20 30]]]
'''
x, y = split_xy5(dataset, 3, 2)
print(x, "\n", y)
print(x.shape)      # (6, 3, 3)
print(y.shape)      # (6, 2, 3)
'''
x :
[[[ 1 11 21]
  [ 2 12 22]
  [ 3 13 23]]

 [[ 2 12 22]
  [ 3 13 23]
  [ 4 14 24]]

 [[ 3 13 23]
  [ 4 14 24]
  [ 5 15 25]]

 [[ 4 14 24]
  [ 5 15 25]
  [ 6 16 26]]

 [[ 5 15 25]
  [ 6 16 26]
  [ 7 17 27]]

 [[ 6 16 26]
  [ 7 17 27]
  [ 8 18 28]]]
y : 
 [[[ 4 14 24]
  [ 5 15 25]]

 [[ 5 15 25]
  [ 6 16 26]]

 [[ 6 16 26]
  [ 7 17 27]]

 [[ 7 17 27]
  [ 8 18 28]]

 [[ 8 18 28]
  [ 9 19 29]]

 [[ 9 19 29]
  [10 20 30]]]
'''

