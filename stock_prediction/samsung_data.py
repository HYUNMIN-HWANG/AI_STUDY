# csv

import numpy as np
import pandas as pd

df = pd.read_csv('./stock_prediction/삼성전자_raw.csv', index_col=0, header=0, encoding='cp949')  
# print(df.shape) # (2400, 14)
# print(df.info())
# print(df.columns)
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
    #    '외인(수량)', '외국계', '프로그램', '외인비'],

# 특수기호 제거
df['시가'] = df['시가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['고가'] = df['고가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['저가'] = df['저가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['종가'] = df['종가'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['거래량'] = df['거래량'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['금액(백만)'] = df['금액(백만)'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['개인'] = df['개인'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['기관'] = df['기관'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['외인(수량)'] = df['외인(수량)'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['외국계'] = df['외국계'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
df['프로그램'] = df['프로그램'].str.replace(pat=r'[^\w]', repl=r'', regex=True)
# print(df)

# 문자형을 숫자로 변환
for col in df.columns :
    df[col] = pd.to_numeric(df[col])
# print(df.info()) 


# 1. 일자를 기준으로 오름차순
df_sorted = df.sort_values(by='일자' ,ascending=True) 
print(df_sorted)

# 2. 예측하고자 하는 값을 맨 뒤로 보낸다.
y = df_sorted.iloc[:,3:4]
# print(y)
del df_sorted['종가']
df_sorted['종가'] = y 
print(df_sorted)
print(df_sorted.columns)

# 3. 결측값이 들어있는 행 전체 제거
print(df_sorted.isnull().sum())    
# null : 2018-04-30, 2018-05-02, 2018-05-03 >> 거래량  3 / 금액(백만) 3
df_dop_null = df_sorted.dropna(axis=0)
print(df_dop_null.shape)    # (2397, 14)

# 4. 2018-05-04부터 데이터 사용하기 (행 제거)
df_slicing = df_dop_null.iloc[1735:,:]
print(df_slicing)           # [662 rows x 14 columns]
print(df_slicing.shape)   

# 5. 분석하고자 하는 칼럼만 남기기 (열 제거)
# 시가, 고가, 저가, 종가, 거래량, 금액, 신용비, 외인비 (음수 기호있는 칼럼들을 뺐다.)
del df_slicing['등락률']
del df_slicing['개인']
del df_slicing['기관']
del df_slicing['외인(수량)']
del df_slicing['외국계']
del df_slicing['프로그램']

print(df_slicing)   # [662 rows x 8 columns]

# 최종 데이터 확인 
print(df_slicing.shape) # (662, 8)
# print(df_slicing.info()) 
'''
<class 'pandas.core.frame.DataFrame'>
Index: 662 entries, 2018-05-04 to 2021-01-13
Data columns (total 8 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   시가      662 non-null    int64
 1   고가      662 non-null    int64
 2   저가      662 non-null    int64
 3   거래량     662 non-null    float64
 4   금액(백만)  662 non-null    float64
 5   신용비     662 non-null    float64
 6   외인비     662 non-null    float64
 7   종가      662 non-null    int64
dtypes: float64(4), int64(4)
memory usage: 46.5+ KB
None
'''

# numpy 저장 
final_data = df_slicing.to_numpy()
print(final_data)
print(type(final_data)) # <class 'numpy.ndarray'>
print(final_data.shape) # (662, 8)
np.save('./stock_prediction/samsung_slicing_data1.npy', arr=final_data)
