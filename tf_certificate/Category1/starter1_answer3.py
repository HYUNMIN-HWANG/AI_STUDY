import numpy as np

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    # YOUR CODE HERE
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    model.add(Dense(64, input_dim=1))
    model.add(Dense(32))
    model.add(Dense(32))
    model.add(Dense(32))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    model.fit(xs, ys, epochs=100, batch_size=1)

    print(model.predict([10.0]))

    return model

solution_model()