####################################
#Code Multi-class classification using KERAS, train on newswire data set
########################################################################



############################################################
############################################################

### Load the dataset
### vectorize the training and testing data
### the output variables is multi-class (46 topics)
### output label will be clssified by categoriacal variables
### Multi-class single-label classification

############################################################
############################################################


from keras.datasets import reuters

# Load IMDB dataset, the traning and testing set is defaulted
# only keep top 10,000 most frequently occurring words 

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)



### One-hot encoding for the training and testing labels
### classify Reuters newswires into 46 mutually exclusive topics

def to_one_hot(labels, dimension=46):
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

############################################################
############################################################

# Model evaluation by results and plots
# Plot for the training and validating loss
# Plot for the training and validating accuracy
# Plot for the training and validating AUC
# fit the test data and evaluate the loss and accuracy

############################################################
############################################################

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
auc = history.history['auc']
val_auc = history.history['val_auc']
epochs = range(1, len(loss) + 1)


# Plot for the training and validating loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot for the training and validating AUC

plt.clf()
plt.plot(epochs, auc, 'b', label='Training AUC')
plt.plot(epochs, val_auc, 'r', label='Validation AUC')
plt.title('Training and validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()


# Plot for the training and validating accuracy
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.fit(x_train, one_hot_train_labels, epochs=20, batch_size=512)
results = model.evaluate(x_test, one_hot_test_labels)
print("Loss and Accuracy:",results)
