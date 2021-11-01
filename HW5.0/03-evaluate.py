#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

# have the load the dataset again for the test data

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

##### Load the models and print 


##########################################
###################("1D-CNN")#############
##########################################

print("--------------------------------------")
print("1D-CNN training, tesing and validation")  
print("--------------------------------------")

title = "1D-CNN"
from keras.models import load_model
model = load_model('1DCNN.h5')
model.fit(X_train, train_labels, epochs=10,batch_size=512, validation_data=(X_val,val_labels),verbose=1)
print(title+": TEST METRIC (loss,accuracy,AUC):",model.evaluate(X_test,test_labels,batch_size=512,verbose=1))

#############################################
###################("SimpleRNN")#############
#############################################

print("------------------------------------------")
print("SimpleRNN training, tesing and validation")  
print("------------------------------------------")

title = "SimpleRNN"
from keras.models import load_model
model = load_model('SimpleRNN.h5')
model.fit(X_train, train_labels, epochs=10,batch_size=512, validation_data=(X_val,val_labels),verbose=1)
print(title+": TEST METRIC (loss,accuracy,AUC):",model.evaluate(X_test,test_labels,batch_size=512,verbose=1))
