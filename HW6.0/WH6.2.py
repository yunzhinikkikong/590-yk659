#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers

from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets.mnist import load_data



#GET mnist DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000,28,28,1); 
#MODEL
input_img = Input(shape=(28, 28, 1))

##########
# encoding
##########
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
##########
# decoding
##########

x = layers.Conv2D(8, (4, 4), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)

decoded= layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)

model = Model(input_img, decoded)

model.summary()


#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')

# define error threshold
#threshold = 4 * model.evaluate(X,X,batch_size=X.shape[0])

history = model.fit(X, X, epochs=10, batch_size=500,validation_split=0.2)



#PLOT HISTORY
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

plt.title("MNIST-- deep feed forward AE")
plt.legend()
plt.show()



#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(X) 

#RESHAPE
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])


#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.show()




#COMPARE ORIGINAL 


# import fashion_mnist dataset

from keras.datasets import fashion_mnist
(x, y), (x_test, y_test) = fashion_mnist.load_data()


#NORMALIZE AND RESHAPE
x=x/np.max(x) 
x=x.reshape(60000,28,28,1); 
x1= model.predict(x) 

#x1.history = model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)
#model.evaluate(x1,x1,batch_size=x1.shape[0])

x=x.reshape(60000,28,28); #print(X[0])
x1=x1.reshape(60000,28,28);

f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(x[I1])
ax[1].imshow(x1[I1])
ax[2].imshow(x[I2])
ax[3].imshow(x1[I2])
plt.show()

