# csv
# 코덱스 데이터 중 필요한 부분만 남기고 정리하기
# 넘파이로 저장


import pandas as pd
import numpy as np

...
경로 수정해야 함
...

df_kod = pd.read_csv('/content/drive/My Drive/인공지능 과정/stock_prediction/KODEX 코스닥150 선물인버스.csv', index_col=0, header=0, encoding='cp949')   
# print(df_kod)          # 2021/01/15 ~ 2016/08/10
# print(df_kod.shape)    # (1088, 16)
# print(df_kod.columns)
#Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
    #    '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

# print(df2.info())

# 삼성전자 시가와의 상관관계
# 상관계수 -0.69 : 시가, 고가, 저가, 종가
# 상관계수 0.49 : 거래량
# 상관계수 0.57 : 신용비

# 1. 특수기호 제거
df_kod.replace(',','',inplace=True, regex=True)
# print(df_kod)

# 2. 분석하고자 하는 칼럼만 남기기 (열 제거)
# 남길 열 :  시가, 고가, 저가, 종가, 거래량, 신용비
# 열 제거 :  등락률, 금액, 개인, 기관, 외인, 외국계, 프로그램
delete_col_kod = df_kod.drop(['전일비','Unnamed: 6','등락률', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램','외인비'], axis=1)
# print(delete_col_kod)           # [1088 rows x 6 columns]
# print(delete_col_kod.columns)
# Index(['시가', '고가', '저가', '종가', '거래량', '신용비'], dtype='object')

# 2. 문자형을 숫자로 변환
for col in delete_col_kod.columns :
    delete_col_kod[col] = pd.to_numeric(delete_col_kod[col])
# print(delete_col_kod.info()) # dtypes: float64(1), int64(5)

# 3. 일자를 기준으로 오름차순
df_sorted_kod = delete_col_kod.sort_values(by='일자' ,ascending=True) 
# print(df_sorted_kod)

# 5. 결측값 확인 > null 없음
# print(df_sorted_kod.isnull().sum())

# 6. 삼성전자 데이터에서 삭제했던 일자 지우기 (2018.04.30~2018.05.03)
df_sorted_kod.drop(['2018/04/30'],axis=0,inplace=True)
df_sorted_kod.drop(['2018/05/02'],axis=0,inplace=True)
df_sorted_kod.drop(['2018/05/03'],axis=0,inplace=True)
# print(df_sorted_kod)  # [1085 rows x 6 columns]

# 6. numpy로 바꾸기
final_kod = df_sorted_kod.to_numpy()
# print(final_kod)
# print(type(final_kod)) # <class 'numpy.ndarray'>
# print(final_kod.shape) # (1085, 6)


# ================================================
# 7. train, test, vali, pred 데이터 분리 

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
dataset_kod = split_x(final_kod,col,size)
# print(dataset_kod)
# print(dataset_kod.shape) # (1080, 6, 6)

#1. DATA
x_kod = dataset_kod[:-2,:,:]
# print(x_kod)
# print(x_kod.shape)  # (1078, 6, 6)

x_pred_kod = dataset_kod[-1:,:]
# print(x_pred_kod)
# print(x_pred_kod.shape) # (1, 6, 6)

# preprocessing
from sklearn.model_selection import train_test_split
x_train_kod, x_test_kod = train_test_split(x_kod, train_size=0.8, shuffle=True, random_state=311)
x_train_kod, x_val_kod = train_test_split(x_train_kod ,train_size=0.8, shuffle=True, random_state=311)
# print(x_train_kod.shape)        # (689, 6, 6)
# print(x_test_kod.shape)         # (216, 6, 6)
# print(x_val_kod.shape)          # (173, 6, 6)

# MinMaxscaler를 하기 위해서 2차원으로 바꿔준다.
x_train_kod = x_train_kod.reshape(x_train_kod.shape[0],size*col)
x_test_kod = x_test_kod.reshape(x_test_kod.shape[0],size*col)
x_val_kod = x_val_kod.reshape(x_val_kod.shape[0],size*col)
x_pred_kod = x_pred_kod.reshape(x_pred_kod.shape[0],size*col)

from sklearn.preprocessing import MinMaxScaler
scaler_kod = MinMaxScaler()
scaler_kod.fit(x_train_kod)
x_train_kod = scaler_kod.transform(x_train_kod)
x_test_kod = scaler_kod.transform(x_test_kod)
x_val_kod = scaler_kod.transform(x_val_kod)
x_pred_kod = scaler_kod.transform(x_pred_kod)

x_train_kod = x_train_kod.reshape(x_train_kod.shape[0],size,col)
x_test_kod = x_test_kod.reshape(x_test_kod.shape[0],size,col)
x_val_kod = x_val_kod.reshape(x_val_kod.shape[0],size,col)
x_pred_kod = x_pred_kod.reshape(x_pred_kod.shape[0], size,col)

print(x_train_kod.shape)        # (689, 6, 6)
print(x_test_kod.shape)         # (216, 6, 6)
print(x_val_kod.shape)          # (173, 6, 6)
print(x_pred_kod.shape)         # (1, 6, 6)


# 넘파이 저장
np.save('./samsung/kodex_slicing_data1.npy',arr=[x_train_sam, x_test_sam, x_val_sam,x_pred_sam])
