# csv
# 1월 14일 데이터 추가하기

import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/삼성전자1_raw.csv', index_col=0, header=0, encoding='cp949')   
# print(df.shape) # (2400, 14)
# print(df.info())
# print(df.columns)
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
    #    '외인(수량)', '외국계', '프로그램', '외인비'],


#############################################################

# 추가데이터 정리
df2 = pd.read_csv('./samsung/삼성전자2_raw.csv', index_col=0, header=0, encoding='cp949') 

# # 열 삭제
df2.drop(['전일비', 'Unnamed: 6'], axis=1, inplace=True)
# print(df2.shape) # (60, 14)
# print(df2.columns)
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
#        '외인(수량)', '외국계', '프로그램', '외인비'],

# # NULL 값 삭제
# print(df2.isnull().sum())    
df2 = df2.dropna(axis=0)
# print(df2.shape)    # (2, 14)
# print(df2.isnull().sum())  # null 제거 확인

# # 데이터 13일 데이터 갱신 + 14일 데이터 추가
df2_0114 = df2.iloc[:1,:]                      # 1월 14일 데이터
df2_0113 = df2.iloc[1:2,:]                     # 1월 13일 데이터
df = df.append(df2_0114, ignore_index=False)   # 14일 데이터 추가
df[0:1] = df2_0113                             # 1월 13일 데이터 갱신
# print(df[0:1])
# print(df.shape)  # (2401, 14)

#############################################################

# 1. 특수기호 제거
df.replace(',','',inplace=True, regex=True)
# print(df)

# 2. 문자형을 숫자로 변환
for col in df.columns :
    df[col] = pd.to_numeric(df[col])
# print(df.info()) # dtypes: float64(5), int64(9)

# 3. 일자를 기준으로 오름차순
df_sorted = df.sort_values(by='일자' ,ascending=True) 
# print(df_sorted)

# 4. 예측하고자 하는 값을 맨 뒤에 추가한다.
zonga = df_sorted.iloc[:,3]
df_target = df_sorted.drop(['종가'], axis=1)
df_target['종가'] = zonga 
# print(df_target)    
# print(df_target.columns)
# Index(['시가', '고가', '저가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)',
    #    '외국계', '프로그램', '외인비', '종가'],
# [2401 rows x 14 columns]

# 5. 결측값이 들어있는 행 전체 제거
# print(df_target.isnull().sum())    
# null : 2018-04-30, 2018-05-02, 2018-05-03 >> 거래량  3 / 금액(백만) 3
df_drop_null = df_target.dropna(axis=0)
# print(df_drop_null.shape)    # (2398, 14)
# print(df_drop_null.isnull().sum())  # null 제거 확인


# 6. 상관계수 확인
# print(df_dop_null.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2, font='Malgun Gothic', rc={'axes.unicode_minus':False}) # 폰트 크기
sns.heatmap(data=df_drop_null.corr(),square=True, annot=True, cbar=True)
# plt.show()
# 상관계수 0.5 이상 : 시가, 고가, 저가, 종가, 거래량, 외인비

# 7. 분석하고자 하는 칼럼만 남기기 (열 제거)
# 남길 열 : 시가, 고가, 저가, 거래량, 외인비, 종가
# 열 제거 :  등락률, 금액, 신용비, 개인, 기관, 외인, 외국계, 프로그램
delete_col = df_drop_null.drop(['등락률', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램'], axis=1)
# print(delete_col)
# print(delete_col.columns)
# Index(['시가', '고가', '저가', '거래량', '외인비', '종가'], dtype='object')
# print(delete_col.shape) # [2398 rows x 6 columns]

# 8. 액면가 조정 
# 시가, 고가, 저가, 종가 : (1~1735) 50으로 나누기
# 거래량 : (1~1735) 50으로 곱하기

# # 시가
a = delete_col.iloc[:1735,:1] / 50
b = delete_col.iloc[1735:,:1]
delete_col['시가'] = pd.concat([a,b])
# print(delete_col['시가'])
print(delete_col['시가'].shape)    # (2398,)

# # 고가
a = delete_col.iloc[:1735,1:2] / 50
b = delete_col.iloc[1735:,1:2]
delete_col['고가'] = pd.concat([a,b])
# print(delete_col['고가'])
# print(delete_col['고가'].shape)    # (2398,)

# # 저가
a = delete_col.iloc[:1735,2:3] / 50
b = delete_col.iloc[1735:,2:3]
delete_col['저가'] = pd.concat([a,b])
# print(delete_col['저가'])
# print(delete_col['저가'].shape)    # (2398,)

# # 거래량
a = delete_col.iloc[:1735,3:4] * 50
b = delete_col.iloc[1735:,3:4]
delete_col['거래량'] = pd.concat([a,b])
# print(delete_col['거래량'])
# print(delete_col['거래량'].shape)    # (2398,)

# # 종가
a = delete_col.iloc[:1735,5:6] / 50
b = delete_col.iloc[1735:,5:6]
delete_col['종가'] = pd.concat([a,b])
# print(delete_col['종가'])
# print(delete_col['종가'].shape)    # (2398,)

# 9. 최종 데이터 확인 
print(delete_col)
print(delete_col.shape) # (2398, 6)
print(delete_col.info()) 

'''
<class 'pandas.core.frame.DataFrame'>
Index: 2398 entries, 2011-04-18 to 2021-01-14
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   시가      2398 non-null   float64
 1   고가      2398 non-null   float64
 2   저가      2398 non-null   float64
 3   거래량     2398 non-null   float64
 4   외인비     2398 non-null   float64
 5   종가      2398 non-null   float64
dtypes: float64(6)
memory usage: 131.1+ KB
None
'''

# 10. numpy 저장 
final_data = delete_col.to_numpy()
print(final_data)
print(type(final_data)) # <class 'numpy.ndarray'>
print(final_data.shape) # (2398, 6)

np.save('./samsung/samsung_slicing_data3.npy', arr=final_data)
