# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    # DATA
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape)  # (60000,) (10000,)

    x_train = x_train/255.
    x_test = x_test/255.

    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

    # Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool1D

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(28,28)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(10, activation='softmax'))
    # model.summary()

    # Compile, Train
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, batch_size=16, epochs=20)

    # Evaluate
    results = model.evaluate(x_test, y_test, batch_size=16)
    print("loss : ", results[0])
    print("acc : ", results[1])
    # loss :  0.38713741302490234
    # acc :  0.8842999935150146

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:\\Study\\tf_certificate\\Category2\\mymodel.h5")
