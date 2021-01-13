# dataset 파일을 numpy로 저장
 
import numpy as np
import pandas as pd

# df = pd.read_csv('../data/csv/iris_sklearn.csv') # -> 문제점 : 인덱스가 데이터로 들어가짐
df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)  
print(df)

print(df.shape) # (150, 5)
print(df.info())    # 다른거는 float64 / target만 int

# pandas -> 넘파이로 바꾼다.

aaa = df.to_numpy()
print(aaa)
print(type(aaa))    # target 값이 int에서 float 형태로 바뀌었다.

bbb = df.values
print(bbb)
print(type(bbb))    # target 값이 int에서 float 형태로 바뀌었다.

# npy 저장
np.save('../data/npy/iris_sklearn.npy', arr=aaa)

# 과제
# 판다스의 loc iloc에 대해 정리하시오. (데이터 분리에 대하여)
