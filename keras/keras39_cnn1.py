# CNN > Conv2D

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), \
    strides=2, padding='same',input_shape=(10, 10, 1)))
        # 필터(=output) 10개 / 
        # kernel size (2,2)로 자른다. / 
        # strides 걷다 / default=1 / =1 : 한 칸씩 간다. =2 : 두 칸씩 뛰어간다. /
        # padding 원래 shape와 동일한 모양.  default = 'valid' / 
        # input_shape=(행, 열, 채널(=색 유무)) (1=gray, 생략가능) /
        # 그 다음 레이어에도 4차원이 전달된다.
        
# model.add(MaxPooling2D(pool_size=3))    # pool_size : Conv2D한 이후에 가능, 필요없는 부분을 제외하고 제일 큰 값만 뽑아낸다.
model.add(MaxPooling2D(pool_size=(2,2)))  # default=2
# model.add(MaxPooling2D(pool_size=(2,3))) 

# model.add(Conv2D(9, (2,2), padding='valid'))
                              # (filter, (kernel size)) 자동으로 인식
# model.add(Conv2D(9, (2,3))) # kernel size 위와 달라도 가능함
# model.add(Conv2D(8, 2))     # (2, 2) = 2  같은 것으로 인식한다. 
                              # 특성을 추축하는 것이기 때문에 Conv2D 여러번 쓰면 더 성능이 좋아질 수 있다. (너무 많이 쓰는 건 또 아님)

model.add(Flatten())    # Flatten : Dense 층과 연결하기 위해서 2차원으로 바꿈
model.add(Dense(1))     # output

model.summary()

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369          >> output 노드 수 : 9(shape) - 2(pool size) + 1
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           296
_________________________________________________________________
flatten (Flatten)            (None, 392)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 393
=================================================================
Total params: 1,108
Trainable params: 1,108
Non-trainable params: 0
_________________________________________________________________

파라미터 개수  = filter * kernel size * channel + bias(filter수 만큼)
(input_dim(channel) * kernel_size + bias(=1)) * output(filter)
"""
