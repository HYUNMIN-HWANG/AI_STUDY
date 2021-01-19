# pandas는 열이 중심
# pandas를 복사해서 원본을 유지하기 위해서는 .copy를 사용한다.

import pandas as pd 

df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[7,8,9,10]],columns=list('abcd'), index=('가','나','다'))
print(df)

df2 = df    # df와는 별도의 df2를 만들고 싶지만, 해당 명령어로는 별도 datagrame이 만들어지지 않음 (주소 동일함)

df2['a'] = 100
print(df2)
print(df)   # 원래 있던 df도 함께 변한다.

# print(id(df), id(df2))  # 2088931984816 2088931984816 : 주소가 동일하다

print("========================")
df3 = df.copy()         # 원본 건드리지 않고 변화시키기 위해서는 .copy를 사용해야 한다.
df2['b'] = 333

print(df)
print(df2)
print(df3)              # df3는 변하지 않음

print("========================")

df = df + 99            # '='가 아닌 부호에서는 카피가 된다.
print(df)
print(df2)