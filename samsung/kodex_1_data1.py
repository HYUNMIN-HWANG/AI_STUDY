import pandas as pd
import numpy as np
df = pd.read_csv('./samsung/KODEX 코스닥150 선물인버스.csv', index_col=0, header=0, encoding='cp949')   
print(df.columns)
#Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
    #    '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],