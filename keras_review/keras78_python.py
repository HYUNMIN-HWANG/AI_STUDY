import numpy as np


# if 문 확인
def oneline (x) :
    return (x>=0)*1 + (x<0)*2
print(oneline(-5))  # 2
print(oneline(5))   # 1

# maximum,  minimum
a = np.array([[0, 1, 6],
             [2, 4, 1]])
b = np.array([4, 2, 9])
c = np.array([1,0,10])

print(np.max(a))    # 6
print(np.max(a,axis=0)) # [2 4 6]
print(np.max(a,axis=1)) # [6 4]
print(np.maximum(b,c))  # [ 4  2 10]