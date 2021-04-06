# Convolutional Neural Network

import tensorflow as tf
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from keras.utils import plot_model
from keras.models import Model
import numpy as np
from keras.layers import LeakyReLU, BatchNormalization, Concatenate, Convolution1D, Dense, MaxPooling1D, AveragePooling1D, Flatten, Input, UpSampling1D
from keras.models import Sequential
from keras.optimizers import SGD
from keras import regularizers



"""
A CLI interface to work with CNN type DNN Echo/Noise Cancellation/Removal.
"""


#********************** Info section ***********************
__author__     = "HAMZA KHEDDAR"
__copyright__  = "2021, MEDEA University"
__email__      = "hamza.kheddar@gmail.com"
#******************** End of Info section ******************

#************************ Constants ************************

VERBOSE=1
VALIDATION_SPLIT=0.2  # The portion of data to use for validation
leaky_relu_alpha=0.1

print('.......................Y_Training loading........................')
Y_Train = np.loadtxt('Clean.txt', delimiter=',')
Y_Train = Y_Train.reshape((Y_Train.shape[0],Y_Train.shape[1],1)) # Output clean speech (Y_Train.shape[0],Y_Train.shape[1],1 to match dimension)

print(Y_Train.ndim)
print(Y_Train.shape)
print('.................Clean Input Speech loaded Successfully........................')

print('.......................X_Training loading........................')
X_Train = np.loadtxt('Noisy.txt', delimiter=',')

X_Train = X_Train.reshape((X_Train.shape[0],X_Train.shape[1],1))
print(X_Train.ndim)
print(X_Train.shape)
print('.................Echo/Noise Input Speech loaded Successfully........................')

print('........................Model Creating.........................')

# For simplicity there are 2 convolutions in each layer
# There are 4 pooling (down sampling) layers for autoencoder
# There are 4 up-samping layers for autodecoder


input_coder =  Input(shape=(256,1))

# Input is 256 or 160 samples, 20 ms for sampling rate of 8 kHz
# Of course speech can be wide-band. 

######################### NOISE SUPPRESSION ###################################

conv1 = Convolution1D(512,3,activation='relu',padding='same',strides=1)(input_coder)
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
upsmp1 = UpSampling1D(size=2)(conv10)


conv11 = Convolution1D(16,3,activation='relu',padding='same',strides=1)(upsmp1)
conv12 = Convolution1D(32,3,activation='relu',padding='same',strides=1)(conv11)
upsmp2 = UpSampling1D(size=2)(conv12)

conv13 = Convolution1D(64,3,activation='relu',padding='same',strides=1)(upsmp2)
conv14 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(conv13)
# Of course, there can be more than 2 (two) convolutions in each layer
upsmp3 = UpSampling1D(size=2)(conv14)

conv15 = Convolution1D(128,3,activation='relu',padding='same',strides=1)(upsmp3)
conv16 = Convolution1D(256,3,activation='relu',padding='same',strides=1)(conv15)
upsmp4 = UpSampling1D(size=2)(conv16)

conv17 = Convolution1D(256,3,activation='relu',padding='same',strides=1)(upsmp4)
conv18 = Convolution1D(256,3,activation='relu',padding='same',strides=1)(conv17)

outputN = Convolution1D(1,1,activation='tanh',padding='same')(conv18)

######################### MAIN OF AUTO-ENCODING ##########################
output = Convolution1D(256,9,padding='same',strides=1)(outputN)
output =LeakyReLU(alpha=leaky_relu_alpha)(output)
output = Convolution1D(256,1,padding='same',strides=1)(output)
output = Convolution1D(256,9, padding='same',strides=1)(output)
output = BatchNormalization()(output)
output = LeakyReLU(alpha=leaky_relu_alpha)(output)
output = MaxPooling1D(pool_size=(2))(output)
############################ INCEPTION BLOCK 01 ##########################
incept1 = Convolution1D(1,1,padding='same')(output)
incept1 =Flatten()(incept1)
incept2 = Convolution1D(1,1,padding='same')(output)
incept2= LeakyReLU(alpha=leaky_relu_alpha)(incept2)
incept2 = Convolution1D(1,3,padding='same')(incept2)
incept2= LeakyReLU(alpha=leaky_relu_alpha)(incept2)
incept2 =Flatten()(incept2)
incept3 = Convolution1D(1,1,padding='same')(output)
incept3= LeakyReLU(alpha=leaky_relu_alpha)(incept3)
incept3 = Convolution1D(1,5,padding='same')(incept3)
incept3= LeakyReLU(alpha=leaky_relu_alpha)(incept3)
incept3 =Flatten()(incept3)
incept4 = MaxPooling1D(pool_size=2, strides=1)(output) # pool size in paper=3
incept4 = Convolution1D(1,1,padding='same')(incept4)
incept4 =Flatten()(incept4)
inception1=Concatenate()([incept4,incept1, incept2, incept3])
inception1 = BatchNormalization()(inception1)
#inception1 = LeakyReLU(alpha=leaky_relu_alpha)(inception1)
#inception1 = MaxPooling1D(pool_size=(2))(inception1)
inception1 = LeakyReLU(alpha=leaky_relu_alpha)(inception1)

inception1 = tf.expand_dims(inception1, axis = -1) # expand dimension along last axis
inception1 = MaxPooling1D(pool_size=(2))(inception1)

############################ INCEPTION BLOCK 02 ##########################
incept5 = Convolution1D(1,1,padding='same')(inception1)
incept5 =Flatten()(incept5)
incept6 = Convolution1D(1,1, padding='same')(inception1)
incept6= LeakyReLU(alpha=leaky_relu_alpha)(incept6)
incept6 = Convolution1D(1,3,padding='same')(incept6)
incept6= LeakyReLU(alpha=leaky_relu_alpha)(incept6)
incept6 =Flatten()(incept6)
incept7 = Convolution1D(1,1,padding='same')(inception1)
incept7= LeakyReLU(alpha=leaky_relu_alpha)(incept7)
incept7 = Convolution1D(1,5,padding='same')(incept7)
incept7= LeakyReLU(alpha=leaky_relu_alpha)(incept7)
incept7 =Flatten()(incept7)
incept8 = MaxPooling1D(pool_size=1)(inception1) # pool size in paper=3
incept8 = Convolution1D(1,1,padding='same')(incept8)
incept8 =Flatten()(incept8)
inception2=Concatenate()([incept5, incept6, incept7,incept8])
inception2 = BatchNormalization()(inception2)
inception2 = LeakyReLU(alpha=leaky_relu_alpha)(inception2)
#inception2 = MaxPooling1D(pool_size=2)(inception2)
inception2 = tf.expand_dims(inception2, axis = -1) # expand dimension along last axis
inception2 = MaxPooling1D(pool_size=(2))(inception2)
############################ INCEPTION BLOCK 03 ##########################
incept9 = Convolution1D(1,1,padding='same')(inception2)
incept9 =Flatten()(incept9)
incept10 = Convolution1D(1,1,padding='same')(inception2)
incept10= LeakyReLU(alpha=leaky_relu_alpha)(incept10)
incept10 = Convolution1D(1,3,padding='same')(incept10)
incept10= LeakyReLU(alpha=leaky_relu_alpha)(incept10)
incept10 =Flatten()(incept10)
incept11 = Convolution1D(1,1,padding='same')(inception2)
incept11= LeakyReLU(alpha=leaky_relu_alpha)(incept11)
incept11 = Convolution1D(1,5,padding='same')(incept11)
incept11= LeakyReLU(alpha=leaky_relu_alpha)(incept11)
incept11 =Flatten()(incept11)
incept12 = MaxPooling1D(pool_size=1)(inception2) # pool size in paper=3
incept12 = Convolution1D(1,1,padding='same')(incept12)
incept12 =Flatten()(incept12)
inception3=Concatenate()([incept9, incept10, incept11,incept12])
inception3 = BatchNormalization()(inception3)
inception3= LeakyReLU(alpha=leaky_relu_alpha)(inception3)
inception3 = tf.expand_dims(inception3, axis = -1) # expand dimension along last axis
inception3 = MaxPooling1D(pool_size=(2))(inception3)

############################ INCEPTION BLOCK 04 ##########################
incept13 = Convolution1D(1,1,padding='same')(inception3)
incept13 =Flatten()(incept13)
incept14 = Convolution1D(1,1,padding='same')(inception3)
incept14= LeakyReLU(alpha=leaky_relu_alpha)(incept14)
incept14 = Convolution1D(1,3,padding='same')(incept14)
incept14= LeakyReLU(alpha=leaky_relu_alpha)(incept14)
incept14 =Flatten()(incept14)
incept15 = Convolution1D(1,1,padding='same')(inception3)
incept15= LeakyReLU(alpha=leaky_relu_alpha)(incept15)
incept15 = Convolution1D(1,5, padding='same')(incept15)
incept15= LeakyReLU(alpha=leaky_relu_alpha)(incept15)
incept15 =Flatten()(incept15)
incept16 = MaxPooling1D(pool_size=1)(inception3) # pool size in paper=3
incept16 = Convolution1D(1,1,padding='same')(incept16)
incept16 =Flatten()(incept16)
inception4=Concatenate()([incept13, incept14, incept15,incept16])
inception4 = BatchNormalization()(inception4)
inception4= LeakyReLU(alpha=leaky_relu_alpha)(inception4)
inception4 = tf.expand_dims(inception4, axis = -1) # expand dimension along last axis
inception4 = MaxPooling1D(pool_size=(2))(inception4)
############################ INCEPTION BLOCK 05 ##########################
incept17 = Convolution1D(1,1,padding='same')(inception4)
incept17 =Flatten()(incept17)
incept18 = Convolution1D(1,1,padding='same')(inception4)
incept18= LeakyReLU(alpha=leaky_relu_alpha)(incept18)
incept18 = Convolution1D(1,3, padding='same')(incept18)
incept18= LeakyReLU(alpha=leaky_relu_alpha)(incept18)
incept18 =Flatten()(incept18)
incept19 = Convolution1D(1,1,padding='same')(inception4)
incept19= LeakyReLU(alpha=leaky_relu_alpha)(incept19)
incept19 = Convolution1D(1,5,padding='same')(incept19)
incept19= LeakyReLU(alpha=leaky_relu_alpha)(incept19)
incept19 =Flatten()(incept19)
incept20 = MaxPooling1D(pool_size=(1))(inception4)  # pool size in paper=3
incept20 = Convolution1D(1,1,padding='same')(incept20)
incept20 =Flatten()(incept20)
inception5=Concatenate()([incept17, incept18, incept19,incept20])
inception5 = BatchNormalization()(inception5)
inception5= LeakyReLU(alpha=leaky_relu_alpha)(inception5)
inception5 = tf.expand_dims(inception5, axis = -1) # expand dimension along last axis
inception5 = MaxPooling1D(pool_size=(2))(inception5)

##########################################################################

output = Convolution1D(20,1,padding='same',strides=1)(inception5)
output_encoder = Convolution1D(20,3, padding='same',strides=1)(output)

############################### MAIN OF AUTO-DECODING #########################
input_decoder = Input(shape=(80,1))
outputd = Convolution1D(20,1,padding='same',strides=1)(input_decoder)
outputd = Convolution1D(20,3, padding='same',strides=1)(outputd)

outputd = LeakyReLU(alpha=leaky_relu_alpha)(outputd)
outputd = UpSampling1D(size=2)(outputd)

############################ INCEPTION BLOCK 01 ##########################
inceptd1 = Convolution1D(64,1,padding='same')(outputd)
inceptd1 =Flatten()(inceptd1)
inceptd2 = Convolution1D(96,1, padding='same')(outputd)
inceptd2= LeakyReLU(alpha=leaky_relu_alpha)(inceptd2)
inceptd2 = Convolution1D(128,3,padding='same')(inceptd2)
inceptd2 =Flatten()(inceptd2)
inceptd3 = Convolution1D(16,1, padding='same')(outputd)
inceptd3 = Convolution1D(32,5, padding='same')(inceptd3)
inceptd3 =Flatten()(inceptd3)
inceptd4 = MaxPooling1D(pool_size=(1))(outputd)  # pool size in paper=3
inceptd4 = Convolution1D(32,1,padding='same')(inceptd4)
inceptd4 =Flatten()(inceptd4)
inceptiond1=Concatenate()([inceptd1, inceptd2, inceptd3,inceptd4])
inceptiond1 = BatchNormalization()(inceptiond1)
inceptiond1 = LeakyReLU(alpha=leaky_relu_alpha)(inceptiond1)
#inceptiond1 = UpSampling1D(size=2)(inceptiond1)
inceptiond1 = tf.expand_dims(inceptiond1, axis = -1) # expand dimension along last axis
inceptiond1 = UpSampling1D(size=2)(inceptiond1)
############################ INCEPTION BLOCK 02 ##########################
inceptd5 = Convolution1D(128,1,padding='same')(inceptiond1)
inceptd5 =Flatten()(inceptd5)
inceptd6 = Convolution1D(128,1,padding='same')(inceptiond1)
inceptd6= LeakyReLU(alpha=leaky_relu_alpha)(inceptd6)
inceptd6 = Convolution1D(192,3,padding='same')(inceptd6)
inceptd6 =Flatten()(inceptd6)
inceptd7 = Convolution1D(32,1, padding='same')(inceptiond1)
inceptd7 = Convolution1D(96,5, padding='same')(inceptd7)
inceptd7 =Flatten()(inceptd7)
inceptd8 = MaxPooling1D(pool_size=(1))(inceptiond1) # pool size in paper=3
inceptd8 = Convolution1D(64,1,padding='same')(inceptd8)
inceptd8 =Flatten()(inceptd8)
inceptiond2=Concatenate()([inceptd5, inceptd6, inceptd7,inceptd8])
inceptiond2 = BatchNormalization()(inceptiond2)
inceptiond2 = LeakyReLU(alpha=leaky_relu_alpha)(inceptiond2)
inceptiond2 = tf.expand_dims(inceptiond2, axis = -1) # expand dimension along last axis
inceptiond2 = UpSampling1D(size=2)(inceptiond2)
############################ INCEPTION BLOCK 03 ##########################
inceptd9 = Convolution1D(192,1,padding='same')(inceptiond2)
inceptd9 =Flatten()(inceptd9)
inceptd10 = Convolution1D(96,1,padding='same')(inceptiond2)
inceptd10 = LeakyReLU(alpha=leaky_relu_alpha)(inceptd10)
inceptd10 = Convolution1D(208,3,padding='same')(inceptd10)
inceptd10 =Flatten()(inceptd10)
inceptd11 = Convolution1D(16,1, padding='same')(inceptiond2)
inceptd11 = Convolution1D(48,5, padding='same')(inceptd11)
inceptd11 =Flatten()(inceptd11)
inceptd12 = MaxPooling1D(pool_size=(1))(inceptiond2) # pool size in paper=3
inceptd12 = Convolution1D(64,1,padding='same')(inceptd12)
inceptd12 =Flatten()(inceptd12)
inceptiond3=Concatenate()([inceptd9, inceptd10, inceptd11,inceptd12])
inceptiond3 = BatchNormalization()(inceptiond3)
inceptiond3 = LeakyReLU(alpha=leaky_relu_alpha)(inceptiond3)
inceptiond3 = tf.expand_dims(inceptiond3, axis = -1) # expand dimension along last axis
inceptiond3 = UpSampling1D(size=2)(inceptiond3)

############################ INCEPTION BLOCK 04 ##########################
inceptd13 = Convolution1D(160,1,padding='same')(inceptiond3)
inceptd13 =Flatten()(inceptd13)
inceptd14 = Convolution1D(112,1,padding='same')(inceptiond3)
inceptd14 = LeakyReLU(alpha=leaky_relu_alpha)(inceptd14)
inceptd14 = Convolution1D(224,3,padding='same')(inceptd14)
inceptd14 =Flatten()(inceptd14)
inceptd15 = Convolution1D(24,1, padding='same')(inceptiond3)
inceptd15 = Convolution1D(64,5, padding='same')(inceptd15)
inceptd15 =Flatten()(inceptd15)
inceptd16 = MaxPooling1D(pool_size=(1))(inceptiond3) # pool size in paper=3
inceptd16 = Convolution1D(64,1,padding='same')(inceptd16)
inceptd16 =Flatten()(inceptd16)
inceptiond4=Concatenate()([inceptd13, inceptd14, inceptd15,inceptd16])
inceptiond4 = BatchNormalization()(inceptiond4)
inceptiond4 = LeakyReLU(alpha=leaky_relu_alpha)(inceptiond4)
inceptiond4 = tf.expand_dims(inceptiond4, axis = -1) # expand dimension along last axis
inceptiond4 = UpSampling1D(size=2)(inceptiond4)

############################ INCEPTION BLOCK 05 ##########################
inceptd17 = Convolution1D(128,1,padding='same')(inceptiond4)
inceptd17 =Flatten()(inceptd17)
inceptd18 = Convolution1D(128,1,padding='same')(inceptiond4)
inceptd18 = LeakyReLU(alpha=leaky_relu_alpha)(inceptd18)
inceptd18 = Convolution1D(256,3,padding='same')(inceptd18)
inceptd18 =Flatten()(inceptd18)
inceptd19 = Convolution1D(24,1, padding='same')(inceptiond4)
inceptd19 = Convolution1D(64,5, padding='same')(inceptd19)
inceptd19 =Flatten()(inceptd19)
inceptd20 = MaxPooling1D(pool_size=(1))(inceptiond4) # pool size in paper=3
inceptd20 = Convolution1D(64,1,padding='same')(inceptd20)
inceptd20 =Flatten()(inceptd20)
inceptiond5=Concatenate()([inceptd13, inceptd14, inceptd15,inceptd16])
inceptiond5 = BatchNormalization()(inceptiond5)
inceptiond5 = LeakyReLU(alpha=leaky_relu_alpha)(inceptiond5)
inceptiond5 = tf.expand_dims(inceptiond5, axis = -1) # expand dimension along last axis
inceptiond5 = UpSampling1D(size=2)(inceptiond5)
##########################################################################
outputd = Convolution1D(128,1,padding='same')(inceptiond5)
outputd = Convolution1D(128,5,padding='same')(outputd)
outputd = UpSampling1D(size=2)(outputd) # added 
outputd = BatchNormalization()(outputd)
outputd = LeakyReLU(alpha=leaky_relu_alpha)(outputd)
output_decoder = Convolution1D(1,1,activation='sigmoid', padding='same')(outputd)

############################# END OF AUTO-ENCODING #######################

# There are infinite amount of parameters selection for each topology
# There are infinite topologies. SOME are well examined in the literature
encoder=Model(inputs=input_coder, outputs=output_encoder)
decoder=Model(inputs=input_decoder, outputs=output_decoder)
#model = Model(inputs=input_coder, outputs=output_decoder)
autoEncoder_outputs = decoder(encoder(input_coder ))
autoEncoder= Model(input_coder, autoEncoder_outputs, name='AE')

print('....................Model Created Sucessfully...................')
print(encoder.summary())
print(decoder.summary())
print(autoEncoder.summary())
#
print('........................Model Compiling.........................')
autoEncoder.compile(loss='mse',optimizer='adam', metrics=['mse'])
print('.................Model Compiled Successfully....................')
#
#
print('........................Model Training.........................')
history = autoEncoder.fit(X_Train,Y_Train,epochs=50,verbose=VERBOSE,validation_split=VALIDATION_SPLIT) # batch_size = 512, epoch = 50
print('.................Model Trained Succesfully.....................')


#save model
model_yaml = autoEncoder.to_yaml()
with open("model.yaml","w") as yaml_file:
    yaml_file.write(model_yaml)
autoEncoder.save_weights("model.h5")
print(".............Saved model to disk successfully......")
