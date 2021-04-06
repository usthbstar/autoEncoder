# Convolutional Neural Network

#********************** Info section ***********************
__author__     = "Hamza kheddar"
__copyright__  = "Medea University"

#******************** End of Info section ******************

#************************ Constants ************************

import tensorflow as tf
import tensorflow

from tensorflow import keras

from keras.utils import plot_model
from keras.models import Model
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Input, UpSampling1D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras import regularizers


VERBOSE=1
VALIDATION_SPLIT=0.15  # The portion of data to use for validation

print('.......................Y_Training loading........................')
Y_Train = np.loadtxt('Clean.txt', delimiter=',')
Y_Train = Y_Train.reshape((Y_Train.shape[0],Y_Train.shape[1],1)) # Output clean speech (Y_Train.shape[0],Y_Train.shape[1],1 to match dimension)

print(Y_Train.ndim)
print(Y_Train.shape)
print('.................Clean Input Speech loaded Successfully........................')

print('.......................X_Training loading........................')
X_Train = np.loadtxt('Clean.txt', delimiter=',')

X_Train = X_Train.reshape((X_Train.shape[0],X_Train.shape[1],1))
print(X_Train.ndim)
print(X_Train.shape)
print('.................Echo/Noise Input Speech loaded Successfully........................')

print('........................Model Creating.........................')

# For simplicity there are 2 convolutions in each layer
# There are 4 pooling (down sampling) layers for autoencoder
# There are 4 up-samping layers for autodecoder
# 4 is arbitrary, determines the deepness of DNN, BUT HAS BE SAME for both encoder & decoder

inpt = Input(shape=(160,1))

# Input is 160 samples, 20 ms for sampling rate of 8 kHz
# Of course speech can be wide-band. One should take care then

conv1 = Convolution1D(512,3,activation='relu',padding='same',strides=1)(inpt)
conv2 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv1)
pool1 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv2)


conv3 = Convolution1D(256,3,activation='relu',padding='same',strides=1)(pool1)
conv4 = Convolution1D(256,3,activation='relu',padding='same',strides=1)(conv3)
pool2 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv4)


conv5 = Convolution1D(256,3,activation='relu',padding='same',strides=1)(pool2)
conv6 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv5)
pool3 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv6)


conv7 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(pool3)
conv8 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(conv7)
pool4 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv8)


conv9 = Convolution1D(32,3,activation='relu',padding='same',strides=1)(pool4)
conv10 = Convolution1D(16,3,activation='relu',padding='same',strides=1)(conv9)
############################# EXTRA 
conv10 = Convolution1D( 8, kernel_size = (3), activation='relu', padding='same')(conv10)
conv10=BatchNormalization()(conv10)
pool4 = MaxPooling1D(pool_size = (4), padding='same')(conv10)
conv10 = Convolution1D( 8, 3, activation='relu', padding='same')(pool4)
conv10 = Convolution1D( 8, 3, activation='relu', padding='same')(conv10)
#############
conv10 = MaxPooling1D(pool_size = (4), strides=4, padding='same')(conv10) 
conv10 = Convolution1D( 4, 3, strides=4, activation='relu', padding='same')(conv10)
conv10 = Convolution1D( 4, 3, strides=4, activation='relu', padding='same')(conv10)
conv10 = MaxPooling1D(pool_size = (2), strides=2, padding='same')(conv10)
conv10 = Convolution1D( 2, 4, strides=2, activation='relu', padding='same')(conv10)
encoder=Model(inputs=inpt, outputs=conv10)
encoder.summary()

input_decoder = Input(shape = (1, 2) )

#############

input_decoder = Input(shape = (1, 2) ) ############# 
upsmp1 = UpSampling1D(size=2)(input_decoder) 
conv11 = Convolution1D( 4, 3, activation='relu', padding='same')(upsmp1) 
upsmp1 = UpSampling1D(size=8)(conv11) 
conv11 = Convolution1D( 8, 3, activation='relu', padding='same')(upsmp1) 
conv12 = Convolution1D( 8, 3, activation='relu', padding='same')(conv11) 
pool4 = UpSampling1D(size=10)(conv12) 
conv10 = Convolution1D( 1, kernel_size = (3), activation='tanh', padding='same')(pool4) 
decoder = Model(inputs=input_decoder, outputs=conv10)
decoder.summary()

autoencoder_outputs = decoder(encoder(inpt))
model= Model(inpt, autoencoder_outputs, name='AE')
model.summary()

print('....................Model Created Sucessfully...................')
print(model.summary())
#
print('........................Model Compiling.........................')
model.compile(loss='mse',optimizer='adam', metrics=['mse'])
print('.................Model Compiled Successfully....................')
#
#
print('........................Model Training.........................')
history = model.fit(X_Train,Y_Train,512,5,verbose=VERBOSE,validation_split=VALIDATION_SPLIT) # batch_size = 512, epoch = 50
print('.................Model Trained Succesfully.....................')

#%%
#save model
model_yaml = model.to_yaml()
with open("model.yaml","w") as yaml_file:
    yaml_file.write(model_yaml)
model.save_weights("model.h5")
print(".............Saved model to disk successfully......")

encoder.save('encoder.h5')
print(".............Saved  encoder model to disk successfully......")
decoder.save('decoder.h5')
print(".............Saved  decoder model to disk successfully......")

#%%

#%%


