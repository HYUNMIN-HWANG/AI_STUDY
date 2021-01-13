# pd.read_csv : csv 불러오기
 
import numpy as np
import pandas as pd

# df = pd.read_csv('../data/csv/iris_sklearn.csv') # -> 문제점 : 인덱스가 데이터로 들어가짐
df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)  

# index_col : 디폴트 None
# index_col=0 : 0번째가 인덱스임

# header : 디폴트 0
# 기본적으로 헤더는 그대로 헤더로 읽힌다. 
# header=None : 헤더가 없을 때

print(df)

'''
     sepal_length  sepal_width  petal_length  petal_width  Target
0             5.1          3.5           1.4          0.2       0
1             4.9          3.0           1.4          0.2       0
2             4.7          3.2           1.3          0.2       0
3             4.6          3.1           1.5          0.2       0
4             5.0          3.6           1.4          0.2       0
..            ...          ...           ...          ...     ...
145           6.7          3.0           5.2          2.3       2
146           6.3          2.5           5.0          1.9       2
147           6.5          3.0           5.2          2.0       2
148           6.2          3.4           5.4          2.3       2
149           5.9          3.0           5.1          1.8       2

[150 rows x 5 columns]
'''
