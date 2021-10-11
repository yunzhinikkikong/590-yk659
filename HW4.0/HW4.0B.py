#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nikkikong
""" 

#############################
#### Loading the dataset ####
#############################



# Training a convent from scratch on a small dataset
# The dataset downloaded from Kaggle has 25,000 images.
# the textbook example downsize the dataset by:
# total training cat images: 1000
# total training dog images: 1000
# total validation cat images: 500
# total validation dog images: 500
# total test cat images: 500
#total test dog images: 500

import os, shutil
original_dataset_dir = 'DOGS-AND-CATS'
base_dir = 'cats_and_dogs_small'
os.mkdir(base_dir)

# Directories for the training, validation, and test splits

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
# Directory with testing cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# Directory with testing dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copies the first 1,000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copies the next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
    
  # Copies the first 1,000 dog images to train_dogs_dir   
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
## double check the count   
print('total training cat images:', len(os.listdir(train_cats_dir)))

print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))

print('total test dog images:', len(os.listdir(test_dogs_dir)))

#######################################################################################
    
#############################
#### Model Building ####
#############################

# a small convnet for dogs vs. cats classification
from keras import layers
from keras import models
model = models.Sequential()

# #### CNN architecture:
# ### a stack of alternated Conv2D (with relu activation) and MaxPooling2D layers.
# ### binary-classification problem, so the final output of network will be 
# ### a single unit (a Dense layer of size 1) and a sigmoid activation.
# ### This unit will encode the probability that the network is looking at one class or the other.

# model.add(layers.Conv2D(32, (3, 3), activation='relu',
# input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# # Configuring the model for training
# ## use binary crossentropy as the loss since it is a binary classification problem
# ## return accuracy
from keras import optimizers
# model.compile(loss='binary_crossentropy',
# optimizer=optimizers.RMSprop(lr=1e-4),
# metrics=['acc'])


# #############################
# #### Data Preprocessing ####
# #############################

# ### Since the data currently sits on a drive as JPEG files, 
# ### ImageDataGenerator will be used to convert the files into pixel values
# ### the steps are as follows:
# # 1 Read the picture files.
# # 2 Decode the JPEG content to RGB grids of pixels.
# # 3 Convert these into floating-point tensors.
# # 4 Rescale the pixel values (between 0 and 255) to the [0, 1] interval since 
#  #NN prefer to deal with small input values).

from keras.preprocessing.image import ImageDataGenerator
# # Rescales all images by 1/255
# train_datagen = ImageDataGenerator(rescale=1./255) 
# test_datagen = ImageDataGenerator(rescale=1./255)
# # for training set
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),  # Resizes all images to 150 × 150
#     batch_size=20,
#     class_mode='binary') # binary_crossentropy loss, so binary labels.
# # same for validation set
# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')


# ### There are 20 samples in each batch (the batch size) with of 150 × 150 RGB
# # images and binary labels (shape (20,))
# for data_batch, labels_batch in train_generator:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break

# #############################
# #### Model Fitting #######
# #############################

# ### Fitting the model using a batch generator
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50)


# # save the model
# model.save('cats_and_dogs_small_1.h5')

# #############################
# #### Model Evaluation #######
# #############################

# ## training and validation loss plot
# ## training and validation accuracy plot

# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()


####################################################################################

###########################################
#### Model Building (with data augmentation)####
###########################################

# Since the above CNN model has overfitting issue, 
# next step is to apply data augmentation

#############################
#### Model Building ####
#############################

###### The CNN architecture is the same as previous model:
### a stack of alternated Conv2D (with relu activation) and MaxPooling2D layers.
### binary-classification problem, so the final output of network will be 
### a single unit (a Dense layer of size 1) and a sigmoid activation.
### This unit will encode the probability that the network is looking at one class or the other.
###### However, this time, it includes a Dropout layer to fight overfitting

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
# a Dropout layer
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Configuring the model for training
## use binary crossentropy as the loss since it is a binary classification problem
## return accuracy
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])


#############################
#### Data Preprocessing ####
#############################

# Setting up a data augmentation configuration via ImageDataGenerator
# The data-augmentation generator will only use for training set

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

## Here I attach some description of the paramenter from the texbook:

### rotation_range is a value in degrees (0–180), a range within which to randomly
#rotate pictures.
### width_shift and height_shift are ranges (as a fraction of total width or
#height) within which to randomly translate pictures vertically or horizontally.
### shear_range is for randomly applying shearing transformations.
### zoom_range is for randomly zooming inside pictures.
### horizontal_flip is for randomly flipping half the images horizontally—relevant
#when there are no assumptions of horizontal asymmetry (for example,real-world pictures).
### fill_mode is the strategy used for filling in newly created pixels, which can
# appear after a rotation or a width/height shift.

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

#### Keep the same converting way for validation set
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

#############################
#### Model Fitting #######
#############################

### Fitting the model using the generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

# Save the model 
model.save('cats_and_dogs_small_2.h5')

#############################
#### Model Evaluation #######
#############################

## training and validation loss plot
## training and validation accuracy plot

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


exit()

############################################
#### Visualizing what convnets learn #######
############################################

### There are three visualization ways mentioned in the textbook

### 1. Visualizing intermediate convnet outputs (intermediate activations)—Useful for
# understanding how successive convnet layers transform their input, and for getting
# a first idea of the meaning of individual convnet filters.
### 2. Visualizing convnets filters—Useful for understanding precisely what visual pattern
# or concept each filter in a convnet is receptive to.
### 3. Visualizing heatmaps of class activation in an image—Useful for understanding
# which parts of an image were identified as belonging to a given class, thus allowing
# you to localize objects in images.

############################################
#### Visualizing intermediate activations ####
############################################

## Visualizing intermediate activations is used when we want to check a certain input's (image's)
## feature maps that output from selected or all convolution and pooling layers in the network.
## This visualization method shows how an input is decomposed into the different filters learned by network

### input image from test data

img_path = 'cats_and_dogs_small/test/cats/cat.1700.jpg'
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
# display the pic

import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()


### Load the second model (with data augmentation)
from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')

### Extracts the outputs of the top eight layers and 
### Creates a model that will return these outputs, given the model input
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)


### Visualizing every channel in every intermediate activation

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    # Tiles the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        # Post-processes the feature to make it visually palatable
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                              :, :,
                                              col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Displays the grid
            display_grid[col * size : (col + 1) * size,
                                       row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


###########################################################################################

# The following two visualization methods are using VGG16 network

############################################
#### Visualizing convnet filters ####
############################################

## The second visualization displays the visual pattern that each filter is meant to respond to. 
## This is done with gradient ascent input space: applying gradient descent to the value of the input image of a convnet so as
## to maximize the response of a specific filter, starting from a blank input image. The
## resulting input image will be one that the chosen filter is maximally responsive to.
## The process is as follow:
# 1. build a loss function that maximizes the value of a given filter in a given convolution layer
# 2. use stochastic gradient descent to adjust the values of the input image 
# so as to maximize this activation value.


import numpy as np
from keras.applications import VGG16
from keras import backend as K
import matplotlib.pyplot as plt

# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    #Normalizes the tensor:
    #centers on 0, ensures that std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    # Clips to [0, 1]
    x = np.clip(x, 0, 1)
    x *= 255
    # Converts to an RGB array
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Function to generate filter visualizations
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Computes the gradient of the input picture with regard to this loss
    grads = K.gradients(loss, model.input)[0]
    # Normalization trick: normalizes the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # Returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    # Starts from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    # Runs gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    return deprocess_image(img)

# try this with filter 0 in layer block3_conv1
plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

# Now, start visualizing every filter in every layer.
# For simplicity, only look at the first 64 filters in each layer, and 
# only look at the first layer of each convolution block:
# (block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1).

# Generating a grid of all filter response patterns in a layer

###################################
### input each layer name here:
layer_name = 'block1_conv1'
###################################
size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
# Iterates over the rows of the results grid
for i in range(8):
    # Iterates over the columns of the results grid
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        # Puts the result in the square (i, j) of the results grid
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
            vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()




##################################################
#### Visualizing heatmaps of class activation ####
##################################################

### The third one is useful for understanding which parts of a given image led a convnet 
# to its final classification decision.

### This general category of techniques is called class activation map (CAM) visualization,
# and it consists of producing heatmaps of class activation over input images.

#### Loading the VGG16 network with pretrained weights

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')

########################################
# Preprocessing an input image for VGG16
########################################
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
# I choosed a dog pic from cat and dog example testing set
img_path = 'cats_and_dogs_small/test/dogs/dog.1700.jpg'

### convert this image into something the VGG16 model can read
# : the model was trained on images of size 224 × 244, so resize it to 224 × 224,
#  convert it to a Numpy float32 tensor, and apply these preprocessing rules:

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
# Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
# Preprocesses the batch (this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

np.argmax(preds[0])

dog_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(dog_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],
[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('dog.jpg', superimposed_img)