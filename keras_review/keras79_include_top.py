from tensorflow.keras.applications import VGG19

model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = True
# model.trainable = False

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

'''
include_top=False
model.trainable = True
=================================================================
Total params: 20,024,384
Trainable params: 20,024,384
Non-trainable params: 0
_________________________________________________________________
32
32
'''

'''
include_top=True
model.trainable = True
=================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
38
38
'''

'''
include_top=False
model.trainable = False
=================================================================
Total params: 20,024,384
Trainable params: 0
Non-trainable params: 20,024,384
_________________________________________________________________
32
0
'''

'''
include_top=True
model.trainable = False
=================================================================
Total params: 143,667,240
Trainable params: 0
Non-trainable params: 143,667,240
_________________________________________________________________
38
0
'''