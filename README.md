# AutoEncoder_ examples


There are two 1D CNN auto-encoders examples, they can be reconfigurable in both input and output according to your compression needs  

Example of CNN Auto-encoder_example01

![](images/6.jpg)

The aove figure depicts the architecture of the encoding part of the CNN auto-encoder used in our scheme. Similarly to the model described in \cite{keles2019deepvocoder}, the encoder network takes speech samples as inputs, with a sampling rate of 8 kHz. The key feature of a CNN is the convolution layer that, in our case, consists of two one-dimensional convolution layers of filter sizes $3\times3$. These convolution layers are conveniently arranged in order to replicate $128\times160$ characteristic vectors. After that, a max-pooling layer is added to the second layer feature vectors in order to downsize the sample number to $128\times80$, while retaining the main features of the input data. The same principle is repeated for the next four groups, with different parameters such as filter size, filter count, stride size and pool size, until we obtain a matrix of size $16\times10$. 

Example of CNN Auto-decoder_example01

![](images/7.jpg)



