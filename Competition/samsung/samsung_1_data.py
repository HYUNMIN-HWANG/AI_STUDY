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

# 2. 예측하고자 하는 값을 맨 뒤에 추가한다.
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

# 4. 액면가 조정 (시가, 고가, 저가, 종가, 거래량, 금액, 개인, 기관, 외인, 외국계, 프로그램)
# # 시가
a = df_dop_null.iloc[:1735,:1] / 50
b = df_dop_null.iloc[1735:,:1]
df_dop_null['시가'] = pd.concat([a,b])
# print(df_dop_null['시가'])
# print(df_dop_null['시가'].shape)    # (2397,)

# # 고가
a = df_dop_null.iloc[:1735,1:2] / 50
b = df_dop_null.iloc[1735:,1:2]
df_dop_null['고가'] = pd.concat([a,b])
# print(df_dop_null['고가'])
# print(df_dop_null['고가'].shape)    # (2397,)

# # 저가
a = df_dop_null.iloc[:1735,2:3] / 50
b = df_dop_null.iloc[1735:,2:3]
df_dop_null['저가'] = pd.concat([a,b])
# print(df_dop_null['저가'])
# print(df_dop_null['저가'].shape)    # (2397,)

# # 거래량
a = df_dop_null.iloc[:1735,4:5] * 50
b = df_dop_null.iloc[1735:,4:5]
df_dop_null['거래량'] = pd.concat([a,b])
# print(df_dop_null['거래량'])
# print(df_dop_null['거래량'].shape)    # (2397,)

# # 종가
a = df_dop_null.iloc[:1735,13:14] / 50
b = df_dop_null.iloc[1735:,13:14]
df_dop_null['종가'] = pd.concat([a,b])
# print(df_dop_null['종가'])
# print(df_dop_null['종가'].shape)    # (2397,)

print(df_dop_null)

# 5. 상관계수 확인
print(df_dop_null.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2) # 폰트 크기
sns.heatmap(data=df_dop_null.corr(),square=True, annot=True, cbar=True)
plt.show()
# 상관계수 0.5 이상 : 시가, 고가, 저가, 종가, 금액, 기관, 외인비

# 6. 분석하고자 하는 칼럼만 남기기 (열 제거)
# 시가, 고가, 저가, 종가, 거래량, 금액, 신용비, 외인비 (음수 기호있는 칼럼들을 뺐다.)
# 음수도 같이 계산해도 되는건가??
del df_dop_null['등락률']
del df_dop_null['거래량']
del df_dop_null['금액(백만)']
del df_dop_null['개인']
del df_dop_null['기관']
del df_dop_null['외인(수량)']
del df_dop_null['외국계']
del df_dop_null['프로그램']

print(df_dop_null)   # [2397 rows x 6 columns]

# 7. 최종 데이터 확인 
print(df_dop_null.shape) # (2397, 6)
# print(df_dop_null.info()) 

'''
<class 'pandas.core.frame.DataFrame'>
Index: 2397 entries, 2011-04-18 to 2021-01-13
Data columns (total 8 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   시가      2397 non-null   float64
 1   고가      2397 non-null   float64
 2   저가      2397 non-null   float64
 3   거래량     2397 non-null   float64
 4   금액(백만)  2397 non-null   float64
 5   신용비     2397 non-null   float64
 6   외인비     2397 non-null   float64
 7   종가      2397 non-null   float64
dtypes: float64(8)
memory usage: 168.5+ KB
None
'''

# numpy 저장 
final_data = df_dop_null.to_numpy()
print(final_data)
print(type(final_data)) # <class 'numpy.ndarray'>
print(final_data.shape) # (2397, 6)
np.save('./stock_prediction/samsung_slicing_data1.npy', arr=final_data)

