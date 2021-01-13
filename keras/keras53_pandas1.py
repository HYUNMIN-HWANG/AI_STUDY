# pandas

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris 

dataset = load_iris()
print(dataset.keys())   # dictionary - key
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values()) # dictionary - value
print(dataset.target_names)     # ['setosa' 'versicolor' 'virginica'] << 0, 1, 2로 표시되어 있다.
print(dataset.feature_names)    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# x = dataset.data
x = dataset['data']
# y = dataset.target
y = dataset['target']

# print(x)
# print(y)
print(x.shape, y.shape) # (150, 4) (150,)
print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# =============== pandas - dataframe (넘파이를 판다스로 바꾼다.) =============== 
df = pd.DataFrame(x, columns=dataset['feature_names']) # 헤더 : 컬럼명을 지정한다.
print(df) 
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   <- 헤더, 컬럼명
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
^
|
인덱스

[150 rows x 4 columns]
'''
print(df.shape) # (150, 4)

print(df.columns) # 헤더명, 칼럼명
# Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
    #    'petal width (cm)'],
    #   dtype='object')

print(df.index) # 따로 명시해주지않으면 0부터 자동 인덱싱 해준다.
# RangeIndex(start=0, stop=150, step=1)

print(df.head()) # dataset의 위에서부터 5개만 출력
'''
df([:5])
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
'''
print(df.tail()) # dataset의 밑에서부터 5개만 출력
'''
df([-5:])
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
'''

print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
                        * Non-Null : null 비어있는 값이 없다. (결측치가 없다.)
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
dtypes: float64(4)
memory usage: 4.8 KB
None
'''

print(df.describe())
'''
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000 <- 150개
mean            5.843333          3.057333           3.758000          1.199333
std             0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000 <- 25% 구간에 값
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000
'''

# 컬럼명 수정
df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
print(df.columns)   # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], dtype='object')
print(df.info())    # 수정된 컬럼명이 반영된다.
print(df.describe())

# X 데이터에 Y 칼람을 추가한다.
# print(df['sepal_length'])
df['Target'] = dataset.target # (150,) 데이터들을 df의 'Target' 칼럼에 넣는다.
print(df.head())
'''
   sepal_length  sepal_width  petal_length  petal_width  Target
0           5.1          3.5           1.4          0.2       0
1           4.9          3.0           1.4          0.2       0
2           4.7          3.2           1.3          0.2       0
3           4.6          3.1           1.5          0.2       0
4           5.0          3.6           1.4          0.2       0
'''

print(df.shape)     # (150, 5)
print(df.columns)   # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Target'], dtype='object')
print(df.index)
print(df.tail())

print(df.info())
print(df.isnull())  # 비어 있는 데이터가 있는지
'''
     sepal_length  sepal_width  petal_length  petal_width  Target
0           False        False         False        False   False
1           False        False         False        False   False
2           False        False         False        False   False
3           False        False         False        False   False
4           False        False         False        False   False
..            ...          ...           ...          ...     ...
145         False        False         False        False   False
146         False        False         False        False   False
147         False        False         False        False   False
148         False        False         False        False   False
149         False        False         False        False   False

[150 rows x 5 columns]
'''
print(df.isnull().sum()) # 비어 있는 데이터 개수
'''
[150 rows x 5 columns]
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
Target          0
dtype: int64
'''
print(df.describe())
'''
       sepal_length  sepal_width  petal_length  petal_width      Target
count    150.000000   150.000000    150.000000   150.000000  150.000000
mean       5.843333     3.057333      3.758000     1.199333    1.000000
std        0.828066     0.435866      1.765298     0.762238    0.819232
min        4.300000     2.000000      1.000000     0.100000    0.000000
25%        5.100000     2.800000      1.600000     0.300000    0.000000
50%        5.800000     3.000000      4.350000     1.300000    1.000000
75%        6.400000     3.300000      5.100000     1.800000    2.000000
max        7.900000     4.400000      6.900000     2.500000    2.000000
'''
print(df['Target'].value_counts()) # y값 개수
'''
2    50
1    50
0    50
'''

# 상관계수 : Target에 가장 많이 연관성이 있는 피쳐가 무엇인가
print(df.corr())
'''
              sepal_length  sepal_width  petal_length  petal_width    Target
sepal_length      1.000000    -0.117570      0.871754     0.817941  0.782561
sepal_width      -0.117570     1.000000     -0.428440    -0.366126 -0.426658 *   <- 상관계수가 낮다. (상관관계가 약하다.)
petal_length      0.871754    -0.428440      1.000000     0.962865  0.949035
petal_width       0.817941    -0.366126      0.962865     1.000000  0.956547 *   <- 가장 상관계수가 높다. (상관관계 있다.)
Target            0.782561    -0.426658      0.949035     0.956547  1.000000
* 참고 : 상관관계가 낮은 데이터를 제거하면 더 좋은 결과가 나올 수도 있다.
'''
# 시각화1  : 상관계수 히트맵
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2) # 폰트 크기
# sns.heatmap(data=df.corr(),square=True, annot=True, cbar=True)
    # heatmap : 사각형 형태로 만들겠다.
    # 데이터 : df.corr()
    # square=True : 사각형 형태로 표현
    # annot=True : 글씨를 넣겠다.
    # cbar=True : 옆에 있는 바를 넣겠다.
# plt.show()


# 시각화2 : 도수 분포도('hist'ogram)
# 각 피처마다 데이터의 분포를 보여준다.

plt.figure(figsize=(10,6))

plt.subplot(2, 2, 1) # 2행 2열 그림 중에서 첫 번째
plt.hist(x='sepal_length', data=df) # df에 있는 sepal_length를 x값으로 잡는다.
plt.title('sepal_length')
plt.grid()

plt.subplot(2, 2, 2) 
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2, 2, 3) 
plt.hist(x='petal_length', data=df)
plt.title('petal_length')

plt.subplot(2, 2, 4) 
plt.hist(x='petal_width', data=df)
plt.title('petal_width')

plt.show()
# x : min, max값을 기준으로 수치를 잡는다.
# y : 데이터 개수