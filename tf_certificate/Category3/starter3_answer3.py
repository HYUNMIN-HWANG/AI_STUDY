
import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    # urllib.request.urlretrieve(url, 'rps.zip')
    # local_zip = 'rps.zip'
    # zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/')
    # zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator( # YOUR CODE HERE
                                            rescale=1./255,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            width_shift_range=(-1,1),
                                            height_shift_range=(-1,1),
                                            fill_mode='nearest',
                                            validation_split=0.2
                                        )

    batch = 16
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=batch,
        class_mode='categorical',
        subset='training'
    )# YOUR CODE HERE
    # Found 2016 images belonging to 3 classes.

    test_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        batch_size=batch,
        class_mode='categorical',
        subset='validation'
    ) # Found 504 images belonging to 3 classes.

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 2, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, 2, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit_generator(train_generator, epochs=10, steps_per_epoch=len(train_generator)//batch, validation_data=test_generator, validation_steps=4)

    result = model.evaluate(test_generator)
    print("loss ", result[0])
    print("acc ", result[1])

    return model


if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")