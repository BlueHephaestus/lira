import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

input_shape = [80,145,3]
regularization_rate = 1e-4
loss="categorical_crossentropy"
optimizer = SGD()

model = Sequential()
#input 128,128,3
model.add(Conv2D(32, (7, 7), strides=(2, 2), padding="same", input_shape=input_shape, data_format="channels_last", activation="relu", kernel_regularizer=l2(regularization_rate)))
#input 64,64,32
model.add(MaxPooling2D(data_format="channels_last"))
#input 1,1,128
model.add(Flatten())

#input 1*1*128 = 128
model.add(Dense(2, activation="softmax", kernel_regularizer=l2(regularization_rate)))
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

#Use this to check your layer's input and output shapes, for checking your math / calculations / designs
print "Layer Input -> Output Shapes:"
for layer in model.layers:
    print layer.input_shape, "->", layer.output_shape
