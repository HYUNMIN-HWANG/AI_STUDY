# 시계열 데이터 자르는 함수

import numpy as np

a = np.array(range(1, 11))
size = 5

# (1)
def split_x(seq, size) :
    aaa = []  
    for i in range(len(seq) - size + 1) :           # len(seq) - size + 1 : 반복횟수(= 행의 개수), # size : 열의 개수
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

# (2)
# def split_x(seq, size) :
#     aaa = []  
#     for i in range(len(seq) - size + 1) :       # range(len(seq) - size + 1) : 반복횟수(= 행의 개수), # size : 열의 개수
#         subset = seq[i : (i+size)]
#         aaa.append(subset)
#     print(type(aaa))
#     return np.array(aaa)

dataset = split_x(a, size)  # (6, 5)
dataset2 = split_x(a, 3)    # (8, 3)
dataset3 = split_x(a, 2)    # (9, 2)

print("=========================")
print(dataset)
print(dataset2)
print(dataset3)


'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

[[ 1  2  3]
 [ 2  3  4]
 [ 3  4  5]
 [ 4  5  6]
 [ 5  6  7]
 [ 6  7  8]
 [ 7  8  9]
 
[[ 1  2]
 [ 2  3]
 [ 3  4]
 [ 4  5]
 [ 5  6]
 [ 6  7]
 [ 7  8]
 [ 8  9]
 [ 9 10]]
'''

