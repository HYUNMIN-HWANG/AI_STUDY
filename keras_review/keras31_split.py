import numpy as np
a = np.array(range(1, 11))
size = 5

def split_x(seq, size) :
    aaa = []
    for i in range (len(seq)-size+1) :
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

# [1,2,3,4,5,6,7,8,9,10] / size = 5
dataset = split_x(a, size)

print(dataset)