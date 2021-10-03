###################################################
#####Code Binary classification using KERAS, train on IMDB data set
########################################################################


# Load IMDB dataset, the traning and testing set is defaulted
# only keep top 10,000 most frequently occurring words 

from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
     num_words=10000)



# vectorize the training and testing data

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

### Build the model, use a very small network with two hidden layers, each with 64 units


############################################################
############################################################

# Here I explore different combinations of hyperparameter, 
# and leave the best fit version in the below functions
# I tried three 'relu' activation layers with different input units
# and the out layer is sigmoid activation function that output range from 0 to 1
# Actuallt, the textbook gives the best parameter combination that I have tried
# I tried different activation function and layers, but the results are not as good as initial

############################################################
############################################################

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32,activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# subseting the trainin dataset for a validation set

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# Model compile

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,partial_y_train,epochs=20,
              batch_size=512,validation_data=(x_val, y_val))

# save all fitting results in a dict

history_dict = history.history


############################################################
############################################################

# Model evaluation by results and plots
# Plot for the training and validating loss
# Plot for the training and validating accuracy
# fit the test data and evaluate the loss and accuracy

############################################################
############################################################


import matplotlib.pyplot as plt
# save each keys value in the dict
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)



# Plot for the training and validating loss

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot for the training and validating accuracy

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print("Loss and Accuracy:",results)








