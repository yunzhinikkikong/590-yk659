import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

# load the dataset

with open('noveldataset_dict.pickle', 'rb') as file:
    dataset_dict = pickle.load(file)

X_train = dataset_dict['X_train']
X_val = dataset_dict['X_val']
X_test = dataset_dict['X_test']

train_labels = dataset_dict['train_labels']
val_labels = dataset_dict['val_labels']
test_labels = dataset_dict['test_labels']

# make the lables as catogorical variables with three classes:
# 0: Frankenstein
# 1: PrideandPrejudice
# 2: Odyssey
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)


############################################################
############################################################

### Build the model

### Multi-class single-label classification: softmax output, 
# Loss: categorical_crossentropy

### 1D CNN model

### RNN model, here I will use SimpleRNN

### try L1 and L2 regularization with different value of parameter

############################################################
############################################################


### Borrow from lecture note
#---------------------------
#plotting function
#---------------------------
def report(history,title='',I_PLOT=True):

    print(title+": TEST METRIC (loss,accuracy,AUC):",model.evaluate(X_test,test_labels,batch_size=512,verbose=1))

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')
        
        plt.plot(epochs, history.history['auc'], 'go', label='Training auc')
        plt.plot(epochs, history.history['val_auc'], 'g', label='Validation auc')

        plt.title(title)
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY'+title+'.png')   # save the figure to file
        plt.close()
        




from keras import layers
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers import SimpleRNN


verbose = 1
embedding_dim = 128 #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
max_words = 10000  #DEFINES SIZE OF VOCBULARY TO USE
maxlen = 200 #CUTOFF REVIEWS maxlen 200 WORDS)

print("---------------------------")
print("1D-CNN")  
print("---------------------------")



model = Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
### add L1 regularization
model.add(layers.Dense(64, kernel_regularizer=regularizers.l1(0.001),activation='relu'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc','AUC'])
model.summary()
history = model.fit(X_train, train_labels, epochs=10,batch_size=512, validation_data=(X_val,val_labels),verbose=verbose)

# # save the model
model.save('1DCNN.h5')
# # plot the results
report(history,title="CNN")



print("---------------------------")
print("SimpleRNN")  
print("---------------------------")


model = Sequential() 
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
### add L1 regularization
model.add(layers.Dense(64, kernel_regularizer=regularizers.l1(0.001),activation='relu'))
model.add(layers.SimpleRNN(32)) 
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc','AUC'])
model.summary()

history = model.fit(X_train, train_labels, epochs=10,batch_size=512, validation_data=(X_val,val_labels),verbose=verbose)

# # plot the results
report(history,title="SimpleRNN")
# # save the model
model.save('SimpleRNN.h5')

















