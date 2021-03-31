
import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

    series = np.array(sunspots) # YOUR CODE HERE

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000


    time_train = time[:split_time]# YOUR CODE HERE
    x_train = series[:split_time]# YOUR CODE HERE
    time_valid = time[split_time:]# YOUR CODE HERE
    x_valid = series[split_time:]# YOUR CODE HERE

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000


    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    valid_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([

        # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
        tf.keras.layers.Conv1D(60, 3, padding='same', activation='relu', input_shape=[None,1]),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),

        tf.keras.layers.Dense(1)
    ])
    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    model.fit(train_set, batch_size=16, epochs=50)

    result = model.evaluate(valid_set, batch_size=16)
    print("loss ", result[0])

    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")