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

# 13일 데이터 삭제  
df.drop(['2021-01-13'],axis=0,inplace=True)
# print(df)   # [2399 rows x 14 columns]

#############################################################

# 추가데이터 append
df2 = pd.read_csv('./samsung/삼성전자0115.csv', index_col=0, header=0, encoding='cp949') 
# print(df2.shape) # (80, 16)
# print(df2.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
    #    '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

# # 열 삭제
df2.drop(['전일비', 'Unnamed: 6'], axis=1, inplace=True)
# print(df2.shape) # (80, 14)
# print(df2.columns)
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
    #    '외인(수량)', '외국계', '프로그램', '외인비'],

# # NULL 값 확인 > 널 없음
# print(df2.isnull().sum())  

# # df2 index 이름 변경
df2.rename(index = {'2021/01/13' : '2021-01-13'}, inplace=True)
df2.rename(index = {'2021/01/14' : '2021-01-14'}, inplace=True)
df2.rename(index = {'2021/01/15' : '2021-01-15'}, inplace=True)
# print(df2)

# # df에 13, 14, 15일 데이터 합치기
df2_new = df2.iloc[:3,:]                      # 13, 14, 15일 데이터
df = df.append(df2_new, ignore_index=False)   # 14일 데이터 추가

# print(df)
# print(df.shape)  # (2402, 14)


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
siga = df_sorted.iloc[:,0]      # 예측해야 하는 값 : 시가
# print(siga)
df_sorted['Target'] = siga      # 시가를 Target으로 잡고 열 추가함
# print(df_sorted)    
# print(df_sorted.columns)
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
#        '외인(수량)', '외국계', '프로그램', '외인비', 'Target'],
# [2402 rows x 15 columns]

# 5. 결측값이 들어있는 행 전체 제거
# print(df_sorted.isnull().sum())    
# null : 2018-04-30, 2018-05-02, 2018-05-03 >> 거래량  3개 / 금액(백만) 3개
df_drop_null = df_sorted.dropna(axis=0)
# print(df_drop_null.shape)    # (2399, 15)
# print(df_drop_null.isnull().sum())  # null 제거 확인


# 6. 상관계수 확인
# print(df_dop_null.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
sns.heatmap(data=df_drop_null.corr(),square=True, annot=True, cbar=True)
# plt.show()
# 상관계수 0.5 이상 : 시가, 고가, 저가, 종가, 거래량, 외인비

# 7. 분석하고자 하는 칼럼만 남기기 (열 제거)
# 남길 열 : 시가, 고가, 저가, 종가, 거래량, 외인비, Target
# 열 제거 :  등락률, 금액, 신용비, 개인, 기관, 외인, 외국계, 프로그램
delete_col = df_drop_null.drop(['등락률', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램'], axis=1)
# print(delete_col)           # [2399 rows x 7 columns]
# print(delete_col.columns)
# Index(['시가', '고가', '저가', '종가', '거래량', '외인비', 'Target'], dtype='object')

# 8. 액면가 조정 
# 시가, 고가, 저가, 종가 : (1~1735) 50으로 나누기
# 거래량 : (1~1735) 50으로 곱하기

delete_col.iloc[:1735,:3] = delete_col.iloc[:1735,:3] / 50
b = delete_col.iloc[1735:,:3]
# delete_col['시가'] = pd.concat([a,b])
print(delete_col[:3])


"""
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
# print(delete_col)
# print(delete_col.shape) # (2398, 6)
# print(delete_col.info()) 

"""

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




"""
# ================================================
# 10. train, test, vali, pred 데이터 분리 
final_data = delete_col.to_numpy()
# print(final_data)
# print(type(final_data)) # <class 'numpy.ndarray'>
# print(final_data.shape) # (2398, 6)

# size : 며칠씩 자를 것인지
# col : 열의 개수

def split_x(seq, col,size) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

size = 6
col = 6
dataset = split_x(final_data,col,size)
# print(dataset)
# print(dataset.shape) # (2393, 6, 6)


#1. DATA
x_sam = dataset[:-1,:,:7]
# print(x)
# print(x.shape)  # (2392, 6, 6)

y_sam = dataset[1:,-1:,-1:]
# print(y)
# print(y.shape)  # (2392, 1, 1)


x_pred_sam = dataset[-1:,:,:]
# print(x_pred)
# print(x_pred.shape) # (1, 6, 6)


# preprocessing
from sklearn.model_selection import train_test_split
x_train_sam, x_test_sam, y_train_sam, y_test_sam = train_test_split(x_sam, y_sam, train_size=0.8,\
    shuffle=True, random_state=311)
x_train_sam, x_val_sam, y_train_sam, y_val_sam = train_test_split(x_train_sam, y_train_sam, \
    train_size=0.8, shuffle=True, random_state=311)
# print(x_train.shape)        # (1530, 6, 6)
# print(x_test.shape)         # (479, 6, 6)
# print(x_validation.shape)   # (383, 6, 6)


y_train_sam = y_train_sam.reshape(y_train_sam.shape[0],1)
y_test_sam = y_test_sam.reshape(y_test_sam.shape[0],1)
y_val_sam = y_val_sam.reshape(y_val_sam.shape[0],1)
# print(y_train.shape)        # (1530, 1)
# print(y_test.shape)         # (479, 1)
# print(y_validation.shape)   # (383, 1)

# MinMaxscaler를 하기 위해서 2차원으로 바꿔준다.
x_train_sam = x_train_sam.reshape(x_train_sam.shape[0],size*col)
x_test_sam = x_test_sam.reshape(x_test_sam.shape[0],size*col)
x_val_sam = x_val_sam.reshape(x_val_sam.shape[0],size*col)
x_pred_sam = x_pred_sam.reshape(x_pred_sam.shape[0],size*col)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train_sam)
x_train_sam = scaler.transform(x_train_sam)
x_test_sam = scaler.transform(x_test_sam)
x_val_sam = scaler.transform(x_val_sam)
x_pred_sam = scaler.transform(x_pred_sam)

x_train_sam = x_train_sam.reshape(x_train_sam.shape[0],size,col)
x_test_sam = x_test_sam.reshape(x_test_sam.shape[0],size,col)
x_val_sam = x_val_sam.reshape(x_val_sam.shape[0],size,col)
x_pred_sam = x_pred_sam.reshape(x_pred_sam.shape[0], size,col)

print(x_train_sam.shape)        # (1530, 6, 6)
print(x_test_sam.shape)         # (479, 6, 6)
print(x_val_sam.shape)          # (383, 6, 6)
print(x_pred_sam.shape)         # (1, 6, 6)
print(y_train_sam.shape)        # (1530, 1)
print(y_test_sam.shape)         # (479, 1)
print(y_val_sam.shape)          # (383, 1)


# 넘파이 저장
np.save('./samsung/samsung_slicing_data5.npy',\
    arr=[x_train_sam, x_test_sam, x_val_sam, y_train_sam, y_test_sam, y_val_sam, x_pred_sam])
"""