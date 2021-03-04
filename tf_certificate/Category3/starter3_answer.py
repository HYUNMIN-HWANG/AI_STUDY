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
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    '''
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'C:\\Study\\tf_certificate\\Category3\\rps.zip')
    local_zip = 'C:\\Study\\tf_certificate\\Category3\\rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    # zip_ref.extractall('tmp/')
    zip_ref.extractall('C:\\Study\\tf_certificate\\Category3\\tmp\\')
    zip_ref.close()
    '''


    TRAINING_DIR = "C:\\Study\\tf_certificate\\Category3\\tmp\\rps\\"
    training_datagen = ImageDataGenerator(
                    # YOUR CODE HERE)
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
        target_size=(150,150),
        batch_size=batch,
        class_mode='categorical',
        subset="training"
    )# Found 2016 images belonging to 3 classes.

    valid_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        batch_size=batch,
        class_mode='categorical',
        subset="validation"
    )# Found 504 images belonging to 3 classes.

    print(train_generator[0][0].shape, valid_generator[0][0].shape)  # (16, 150, 150, 3) (16, 150, 150, 3)

    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(3,3),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(3,3),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(3,3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    # model.summary()

    # Compile, Train    
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit_generator(train_generator, steps_per_epoch=len(train_generator)//batch, epochs=1000, \
        validation_data=valid_generator, validation_steps=4, callbacks=[lr, es])
    
    
    results = model.evaluate(valid_generator)
    print("loss : ", results[0])
    print("acc : ", results[1])

    # loss :  0.3974750339984894
    # acc :  0.8551587462425232
    
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:\\Study\\tf_certificate\\Category3\\mymodel.h5")
