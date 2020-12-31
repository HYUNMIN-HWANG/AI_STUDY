# 중요함
# model.summary() 에 대하여
# parameter 개수에 대하여

import numpy as np
import tensorflow as tf 

#1. (정제된) 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # y=ax+b를 구성하는 가장 기본 

model = Sequential() #순차적인 모델을 만들 것이다. #윗단의 아웃풋은 아래의 인풋이 됨
model.add(Dense(6, input_dim=1, activation='linear')) # add 모델을 더해나간다. (노드 개수 아웃풋 다섯개 , 인풋 한 개, 선형)
model.add(Dense(7, activation='linear'))   #(위 레이어 노드가 인풋이 된다.)
model.add(Dense(9))
model.add(Dense(10, name='layer1'))
model.add(Dense(10, name='layer2'))
model.add(Dense(10, name='layer1'))  # ValueError: All layers added to a Sequential model should have unique names.
model.add(Dense(1)) #아웃풋 1개

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #  (연산의 개수)
=================================================================
dense (Dense)                (None, 5)                 10
                             (행무시, 5)                          
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49  (<--연산의 총합)
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
* parameter 개수 : bias를 node의 개수에 포함시켜서 다음 layer의 node 개수와 곱한다.
(bias는 모든 파라미터 연산에 포함되기 때문에 노드 개수에 +1 해서 계산하면 됨)
'''

# 실습2 + 과제
# ensemble1, 2, 3, 4에 대해 summary를 계산하고 이해한 것을 과제로 제출할 것
# ex) 모델1과 모델2를 왔다갔다 함

# layer를 만들 때 'name' 에 대해 확인하고 설명할 것 (레이어의 이름)
# layer name을 알아야 하는 이유를 찾아라/ layer 를 반드시 써야할 때가 언제인지 말해라/-이름이 충돌되는 경우, 어떨 때 충돌이 되는지? 찾아라
