#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import seaborn as sns
from keras import losses
import matplotlib.pyplot as plt
from keras import models
from keras import layers

from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets.mnist import load_data



#GET cifar10 DATASET
from keras.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X_test=X_test/np.max(X_test) 

#MODEL
input_img = Input(shape=(32, 32, 3))

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

decoded= layers.Conv2D(3, (3, 3), activation='linear', padding='same')(x)

model = Model(input_img, decoded)

model.summary()


#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')



history = model.fit(X, X, epochs=20, batch_size=500,validation_split=0.2)

# save model

model.save('cifar10CnnAE.h5')

# reported test accuracy after training
print("TEST METRIC (loss) after training:",model.evaluate(X_test,X_test,batch_size=500))

# define error threshold

threshold = 4 * model.evaluate(X,X,batch_size=X.shape[0])

# anomaly detection
# error > threshold == > anomaly

# 1 = anomaly, 0 = normal
X1=model.predict(X) 
X=X.reshape(50000,32*32*3); #print(X[0])
X1=X1.reshape(50000,32*32*3);
errors = losses.mse(X1, X)
anomaly = pd.Series(errors) > threshold
preds = anomaly.map(lambda x: 1.0 if x == True else 0)

print("anomaly fraction (“trained” data):", preds.sum()/X.shape[0],"\n","anomaly count(“trained” data):",preds.sum())


# Plot original and RECONSTRUCTED 

#RESHAPE
X=X.reshape(50000,32,32,3); #print(X[0])
X1=X1.reshape(50000,32,32,3); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.show()

#PLOT HISTORY
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

plt.title("cifar10-- Convolutional AE")
plt.legend()
plt.show()


# import cifar100 dataset

from keras.datasets import cifar100
(x, y), (x_test, y_test) = cifar100.load_data()

# remove “trucks” from CIFAR100
truck_list = np.where(y == 93)[0]
x = np.delete(x, truck_list, axis=0)


#NORMALIZE AND RESHAPE
x=x/np.max(x) 


x1= model.predict(x) 
x=x.reshape(49500,32*32*3); #print(X[0])
x1=x1.reshape(49500,32*32*3);
errors = losses.mse(x1, x)

# 1 = anomaly, 0 = normal
anomaly_mask = pd.Series(errors) > threshold
preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0)


print("anomaly fraction (“anomalies” data):", preds.sum()/x.shape[0],"\n","anomaly count(“anomalies” data):", preds.sum())



x=x.reshape(49500,32,32,3); #print(X[0])
x1=x1.reshape(49500,32,32,3);

f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(x[I1])
ax[1].imshow(x1[I1])
ax[2].imshow(x[I2])
ax[3].imshow(x1[I2])
plt.show()

