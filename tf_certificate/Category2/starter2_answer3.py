import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

    model = Sequential()
    model.add(Conv1D(128, 3, activation='relu', padding='same', input_shape=(28, 28)))
    model.add(Conv1D(128, 3, activation='relu'))

    model.add(Conv1D(64, 2, activation='relu'))
    model.add(Conv1D(64, 2, activation='relu'))

    model.add(Conv1D(32, 2, activation='relu'))
    model.add(Conv1D(32, 2, activation='relu'))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    # model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, epochs=5, batch_size=8)

    result = model.evaluate(x_test, y_test, batch_size=8)
    print("loss", result[0])
    print("acc", result[1])

    # YOUR CODE HERE
    return model


solution_model()