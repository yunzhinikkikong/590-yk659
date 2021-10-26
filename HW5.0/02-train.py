
import numpy as np

with open('noveldataset_dict.pickle', 'rb') as file:
    dataset_dict = pickle.load(file)
### One-hot encoding for the training and testing labels
### classify Reuters newswires into 3 mutually exclusive novels

def to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

############################################################
############################################################

### Build the model

### Multi-class single-label classification: softmax output, 
# Loss: categorical_crossentropy

# Here I explore different combinations of hyperparameter, 
# and leave the best fit version in the below functions
# I tried three 'selu' activation layers with different 64 input units
# 'selu' is one of the Relu alternative
# and the out layer is softmax activation function since it is a Multi-class single-label classification quesion
# The validation/training loss are lower than textbook example

############################################################
############################################################

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='selu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='selu'))
### multi-class output (46 mutually exclusive topics)
model.add(layers.Dense(64, activation='selu'))
model.add(layers.Dense(46, activation='softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc','AUC'])


# subseting the trainin dataset for a validation set


x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]



history = model.fit(partial_x_train,
                   partial_y_train,
                   epochs=40,
                   batch_size=512,
                   validation_data=(x_val, y_val))
