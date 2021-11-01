# -*- coding: utf-8 -*-

import numpy as np
import nltk
import string
nltk.download('punkt')


Frankenstein = open('Frankenstein.txt')
PrideandPrejudice = open('PrideandPrejudice.txt')
Odyssey = open('The_Odyssey.txt')

def read_novel(f):
  # read .txt files
  novel = f.read()
  # remove all "\n"
  novel=novel.replace("\n", " ")
  # turn novel in to chunk of text
  novel_list = nltk.tokenize.sent_tokenize(novel)
  # remove punctutation in every chunk
  novel_list = [''.join(c for c in s if c not in string.punctuation) for s in novel_list]
  return novel_list

Frankenstein_list=read_novel(Frankenstein)
PrideandPrejudice_list=read_novel(PrideandPrejudice)
Odyssey_list=read_novel(Odyssey)

len(Frankenstein_list)

len(PrideandPrejudice_list)

len(Odyssey_list)

### add them together

data_list = Frankenstein_list+PrideandPrejudice_list+Odyssey_list

len(data_list)

# Using Keras for tokenization

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data_list)
sequences = tokenizer.texts_to_sequences(data_list)
one_hot_results = tokenizer.texts_to_matrix(data_list, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
maxlen = 200 #CUTOFF REVIEWS maxlen 200 WORDS)
from keras.preprocessing.sequence import pad_sequences
data= pad_sequences(sequences,maxlen = maxlen)



# labels: novel names
# 0: Frankenstein
# 1: PrideandPrejudice
# 2: Odyssey
labels=np.concatenate((np.repeat(0, 3196), np.repeat(1, 4778),np.repeat(2, 3854)), axis=None)

labels.shape

#------------------------
#PARTITION DATA
#------------------------
#TRAINING: 	 DATA THE OPTIMIZER "SEES"
#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)

f_train=0.8; f_val=0.15; f_test=0.05;

if(f_train+f_val+f_test != 1.0):
	raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

#PARTITION DATA
np.random.seed(1) #set seed
rand_indices = np.random.permutation(data.shape[0])
CUT1=int(f_train*data.shape[0]); 
CUT2=int((f_train+f_val)*data.shape[0]); 
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print("test_idx shape:" ,test_idx.shape)

X_train = data[train_idx]
X_val = data[val_idx]
X_test = data[test_idx]

labels_train = labels[train_idx]
labels_val = labels[val_idx]
labels_test = labels[test_idx]

import pickle

dataset_dict = {"X_train": X_train, "X_test": X_test, "X_val": X_val, "train_labels": labels_train, "test_labels": labels_test, "val_labels": labels_val}

with open('noveldataset_dict.pickle', 'wb') as file:
    pickle.dump(dataset_dict, file)

exit()










