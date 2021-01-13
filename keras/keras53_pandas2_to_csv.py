# to_csv : 
# pandas -> csv로 저장

# pandas

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris 

dataset = load_iris()

# x = dataset.data
x = dataset['data']
# y = dataset.target
y = dataset['target']

# df = pd.DataFrame(x, columns=dataset.feature_names) # 헤더 : 컬럼명을 지정한다.
df = pd.DataFrame(x, columns=dataset['feature_names']) # 헤더 : 컬럼명을 지정한다.

# 컬럼명 수정
df.columns = ['sepal_length','sepal_width','petal_length','petal_width']

# DataFrame에 Y 칼람을 추가한다.
# print(df['sepal_length'])
df['Target'] = y

# pandas 데이터프레임을 csv로 만들겠다.
# 인덱스와 컬럼명까지 같이 저장된다.
df.to_csv('../data/csv/iris_sklearn.csv', sep=',')  # 구분 ','
