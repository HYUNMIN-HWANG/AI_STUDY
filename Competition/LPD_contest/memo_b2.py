mobile = EfficientNetB2(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
mobile.trainable = True
a = mobile.output
a = GlobalAveragePooling2D()(a)
a = Flatten()(a)
a = Dense(4048, activation= 'swish')(a)
a = Dropout(0.3)(a)
a = Dense(1000, activation= 'softmax')(a)

model = Model(inputs = mobile.input, outputs = a)
optimizer=SGD(lr=0.1)