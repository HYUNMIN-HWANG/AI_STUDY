import numpy as np
import csv
import tensorflow as tf
import numpy as np
import urllib
'''
window_size = 30
batch_size = 32
shuffle_buffer_size = 100

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)                    #  차원을 늘린다. [2,2,2] -> [2,2,2,1]
    ds = tf.data.Dataset.from_tensor_slices(series)             # dataset을 만든다.
    print(ds)                                                   # <TensorSliceDataset shapes: (1,), types: tf.int32>
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)   
    # 시계열 데이터를 만들어준다. /  window_size + 1 : 몇 개씩 자를 것인가 / shift : 몇 개씩 뛸 것인가 / drop_reminder=True : 끝까지 데이터셋 길이를 맞춰준다.
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))        # dataset.window : 리스트로 만들어줌 
    ds = ds.shuffle(shuffle_buffer)                             # 데이터 shuffle
    ds = ds.map(lambda w: (w[:-1], w[1:]))                      # x와 y를 분리시켜줌
    return ds.batch(batch_size).prefetch(1)                     # batch size 만큼 통으로 자른다.

x_train = np.array(range(10))
# print(x_train)
train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
print(train_set)
# <PrefetchDataset shapes: ((None, None, 1), (None, None, 1)), types: (tf.int32, tf.int32)>
'''

dataset = tf.data.Dataset.range(100)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window : window.batch(31))
# dataset = dataset.shuffle(100)
# dataset = dataset.map(lambda window : (window[:-1], window[1:]))
# dataset = dataset.batch(5).prefetch(1)

# for x in dataset :
    # print(x.numpy(), y.numpy())
    # print(x)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
        print()

print(dataset)