
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE

    with open('sarcasm.json','r') as f :
        datasets = json.load(f)
    for item in datasets :
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    import numpy as np

    x_train = np.array(sentences[:training_size])
    x_test = np.array(sentences[training_size:])
    y_train = np.array(labels[:training_size])
    y_test = np.array(labels[training_size:])


    print(x_train.shape, x_test.shape)  # (20000,) (6709,)
    print(y_train.shape, y_test.shape)  # (20000,) (6709,)

    from tensorflow.keras.preprocessing.text import Tokenizer
    token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    token.fit_on_texts(x_train)
    token.fit_on_texts(x_test)

    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    pad_x_train = pad_sequences(x_train, padding=padding_type, truncating=trunc_type, maxlen=max_length)
    pad_x_test = pad_sequences(x_test, padding=padding_type, truncating=trunc_type, maxlen=max_length)

    print(pad_x_train.shape)            # (20000, 120)
    print(len(np.unique(pad_x_train)))  # 100

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=training_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Dense(16, activation='relu'),
        # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(pad_x_train, y_train, epochs=5, batch_size=8, validation_split=0.2)

    result = model.evaluate(pad_x_test, y_test, batch_size=8)
    print("loss ", result[0])
    print("acc ", result[1])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
