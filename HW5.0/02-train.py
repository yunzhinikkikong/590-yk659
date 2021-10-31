import pickle
import numpy as np

# load the dataset

with open('/Users/nikkkikong/590-yk659/HW5.0/noveldataset_dict.pickle', 'rb') as file:
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


############################################################
############################################################



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
model.add(layers.Conv1D(64, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(64, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(3))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc','AUC'])
model.summary()
history = model.fit(X_train, train_labels, epochs=10,batch_size=512, validation_data=(X_val,val_labels),verbose=verbose)
#report(history,title="CNN")




print("---------------------------")
print("SimpleRNN")  
print("---------------------------")


model = Sequential() 
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.SimpleRNN(32)) 
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc','AUC'])
model.summary()

history = model.fit(X_train, train_labels, epochs=10,batch_size=512, validation_data=(X_val,val_labels),verbose=verbose)
#report(history,title="SimpleRNN")

















from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='selu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='selu'))
### multi-class output (46 mutually exclusive topics)
model.add(layers.Dense(64, activation='selu'))
model.add(layers.Dense(3, activation='softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc','AUC'])



history = model.fit(X_train,
                   one_hot_train_labels,
                   epochs=40,
                   batch_size=512,
                   validation_data=(X_val, one_hot_val_labels))

from keras.layers import Embedding, SimpleRNN
from keras.layers import Dense
model = models.Sequential()
model.add(Embedding(1000, 32))
model.add(SimpleRNN(32))
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(X_train,
                   one_hot_train_labels,
                   epochs=40,
                   batch_size=512,
                   validation_data=(X_val, one_hot_val_labels))



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


model.fit(X_train, one_hot_train_labels, epochs=20, batch_size=512)
results = model.evaluate(X_test, one_hot_test_labels)
print("Loss, Accuracy and AUC:",results)
