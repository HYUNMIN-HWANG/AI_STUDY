# 인자 3개를 넣어서 함수 변경해야 함
import numpy as np

data = np.load('./stock_prediction/samsung_slicing_data1.npy')
print(data)
print(data.shape)   # (662, 8)


def split_x(seq, size) :
    aaa = []  
    for i in range(len(seq) - size + 1) :       # range(len(seq) - size + 1) : 반복횟수(= 행의 개수), # size : 열의 개수
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(data,5)
print(dataset[0])
print(dataset[0].shape) # (5, 8)