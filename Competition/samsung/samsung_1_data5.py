# csv
# 삼성 원본 데이터 + 13~15일 데이터 추가 + 코덱스 데이터와의 상관관계
# 넘파이로 저장

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

# *** df2 데이터를 df에 넣기 ***
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

# *** df 데이터 정리하기
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
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
# sns.heatmap(data=df_drop_null.corr(),square=True, annot=True, cbar=True)
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
# # 시가, 고가, 저가, 종가 : (1~1735) 50으로 나누기
delete_col.iloc[:1735,:4] = delete_col.iloc[:1735,:4] / 50
delete_col.iloc[:1735,6] = delete_col.iloc[:1735,6] / 50

# # 거래량 : (1~1735) 50으로 곱하기
delete_col.iloc[:1735,4] = delete_col.iloc[:1735,4] * 50
# print(delete_col[1732:1738])  # 액면가 확인

# 9. kodex와 분석할 날짜만 남기기 (2016.08.10 ~ 현재)
split_row = delete_col.iloc[1314:,]
# print(split_row)  # [1085 rows x 7 columns]

# 10. 최종 데이터 확인 
# print(split_row)
# print(split_row.shape) # (1085, 7)
# print(split_row.info()) 

'''
<class 'pandas.core.frame.DataFrame'>
Index: 1085 entries, 2016-08-10 to 2021-01-15
Data columns (total 7 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   시가      1085 non-null   float64
 1   고가      1085 non-null   float64
 2   저가      1085 non-null   float64
 3   종가      1085 non-null   float64
 4   거래량     1085 non-null   float64
 5   외인비     1085 non-null   float64
 6   Target  1085 non-null   float64
dtypes: float64(7)
memory usage: 67.8+ KB
None
'''

# 10. 넘파이로 바꾸기
final_data = split_row.to_numpy()
# print(final_data)
# print(type(final_data)) # <class 'numpy.ndarray'>
# print(final_data.shape) # (1085, 7)

# 11. train, test, vali, pred 데이터 분리

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
col = 7
dataset = split_x(final_data,col,size)
# print(dataset)
# print(dataset.shape) # (1080, 6, 7)

#1. DATA
x_sam = dataset[:-2,:,:6]
# print(x_sam)
# print(x_sam.shape)  # (1078, 6, 6)

y_sam = dataset[2:,-2:,-1:]
# print(y_sam)
# print(y_sam.shape)  # (1078, 2, 1)

x_pred_sam = dataset[-1:,:,:-1]
# print(x_pred_sam)
# print(x_pred_sam.shape) # (1, 6, 6)

# preprocessing
from sklearn.model_selection import train_test_split
x_train_sam, x_test_sam, y_train_sam, y_test_sam = train_test_split(x_sam, y_sam, train_size=0.8,\
    shuffle=True, random_state=311)
x_train_sam, x_val_sam, y_train_sam, y_val_sam = train_test_split(x_train_sam, y_train_sam, \
    train_size=0.8, shuffle=True, random_state=311)
# print(x_train_sam.shape)        # (689, 6, 6)
# print(x_test_sam.shape)         # (216, 6, 6)
# print(x_val_sam.shape)          # (173, 6, 6)

y_train_sam = y_train_sam.reshape(y_train_sam.shape[0],2)
y_test_sam = y_test_sam.reshape(y_test_sam.shape[0],2)
y_val_sam = y_val_sam.reshape(y_val_sam.shape[0],2)
# print(y_train_sam.shape)        # (689, 2)
# print(y_test_sam.shape)         # (216, 2)
# print(y_val_sam.shape)          # (173, 2)

# MinMaxscaler를 하기 위해서 2차원으로 바꿔준다.
x_train_sam = x_train_sam.reshape(x_train_sam.shape[0],size*(col-1))
x_test_sam = x_test_sam.reshape(x_test_sam.shape[0],size*(col-1))
x_val_sam = x_val_sam.reshape(x_val_sam.shape[0],size*(col-1))
x_pred_sam = x_pred_sam.reshape(x_pred_sam.shape[0],size*(col-1))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train_sam)
x_train_sam = scaler.transform(x_train_sam)
x_test_sam = scaler.transform(x_test_sam)
x_val_sam = scaler.transform(x_val_sam)
x_pred_sam = scaler.transform(x_pred_sam)

x_train_sam = x_train_sam.reshape(x_train_sam.shape[0],size,(col-1))
x_test_sam = x_test_sam.reshape(x_test_sam.shape[0],size,(col-1))
x_val_sam = x_val_sam.reshape(x_val_sam.shape[0],size,(col-1))
x_pred_sam = x_pred_sam.reshape(x_pred_sam.shape[0], size,(col-1))

print(x_train_sam.shape)        # (689, 6, 6)
print(x_test_sam.shape)         # (216, 6, 6)
print(x_val_sam.shape)          # (173, 6, 6)
print(x_pred_sam.shape)         # (1, 6, 6)
print(y_train_sam.shape)        # (689, 2)
print(y_test_sam.shape)         # (216, 2)
print(y_val_sam.shape)          # (173, 2)

# 넘파이 저장
np.save('./samsung/samsung_slicing_data5.npy',arr=[x_train_sam, x_test_sam, x_val_sam, y_train_sam, y_test_sam, y_val_sam, x_pred_sam])
# 코랩에서 저장안돼서 vscode로 저장함

################################################################################

# Kodex 컬람 중 삼성전자 시가와 관련있는 컬럼이 뭔지 알아보자.
'''
df3 = pd.read_csv('./samsung/KODEX 코스닥150 선물인버스.csv', index_col=0, header=0, encoding='cp949')   
# print(df3.shape) # (1088, 16)
# print(df3.columns)
#Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
    #    '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

df3.drop(['전일비', 'Unnamed: 6'], axis=1, inplace=True)
# print(df3.shape) # (1088, 14)
# print(df3.columns)
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
    #    '외인(수량)', '외국계', '프로그램', '외인비'],

# # NULL 값 확인 > 널 없음
# print(df2.isnull().sum())  

# 1. 특수기호 제거
df3.replace(',','',inplace=True, regex=True)
# print(df3)

# 2. 문자형을 숫자로 변환
for col in df3.columns :
    df3[col] = pd.to_numeric(df3[col])
# print(df3.info()) # dtypes: float64(3), int64(11)

# 3. 일자를 기준으로 오름차순
df_sorted_kod = df3.sort_values(by='일자' ,ascending=True) 
# print(df_sorted_kod)

# 4.  예측하고자 하는 값을 맨 뒤에 추가한다.
siga2 = df_sorted.iloc[1314:,0]    # 예측해야 하는 값 : 시가 (2016-08-10 부터) 1088개
siga2[:424] = siga2[:424]/50       # 시가 2016-08-10 ~ 2018-05-03 까지 50으로 나누기
siga2 = siga2.values
# print(siga2)

df_sorted_kod['Target'] = siga2
# print(df_sorted_kod) 
# print(df_sorted_kod.columns)
# [1088 rows x 15 columns]
# Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
#        '외인(수량)', '외국계', '프로그램', '외인비', 'Target'],
#       dtype='object'

# 5. 상관계수 확인
# print(df_sorted_kod.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
sns.heatmap(data=df_sorted_kod.corr(),square=True, annot=True, cbar=True)
plt.show()
# 상관계수 -0.69 : 시가, 고가, 저가, 종가
# 상관계수 0.49 : 거래량
# 상관계수 0.57 : 신용비
'''