# 인자 3개를 넣어서 함수 변경해야 함
import numpy as np

data = np.load('./stock_prediction/samsung_slicing_data1.npy')
# print(data)
print(data.shape)   # (2397, 6)

# size : 며칠씩 자를 것인지
# col : 열의 개수

def split_x(seq, size, col) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    print(type(dataset))
    return np.array(dataset)

dataset = split_x(data,5, 8)
print(dataset)
# print(dataset.shape) # (2393, 5, 6)
