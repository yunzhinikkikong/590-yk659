# -*- coding: utf-8 -*-


import nltk
import string
nltk.download('punkt')

Frankenstein = open('/content/Frankenstein.txt')
PrideandPrejudice = open('/content/PrideandPrejudice.txt')
Odyssey = open('/content/The_Odyssey.txt')

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

data_list = Frankenstein_list+PrideandPrejudice_list+Odyssey_list

len(data_list)

# Using Keras for word-level one-hot encoding
def onehot(samples):
  from keras.preprocessing.text import Tokenizer
  tokenizer = Tokenizer(num_words=1000)
  tokenizer.fit_on_texts(samples)
  sequences = tokenizer.texts_to_sequences(samples)
  one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))
  return one_hot_results
# tokenize three novel list
data_onehot=onehot(data_list)

data_onehot.shape

data_onehot

# labels: novel names
# 1: Frankenstein
# 2: PrideandPrejudice
# 3: Odyssey
labels=np.concatenate((np.repeat(1, 3196), np.repeat(2, 4778),np.repeat(3, 3854)), axis=None)

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
rand_indices = np.random.permutation(data_onehot.shape[0])
CUT1=int(f_train*data_onehot.shape[0]); 
CUT2=int((f_train+f_val)*data_onehot.shape[0]); 
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print("test_idx shape:" ,test_idx.shape)

X_train = data_onehot[train_idx]
X_val = data_onehot[val_idx]
X_test = data_onehot[test_idx]

labels_train = labels[train_idx]
labels_val = labels[val_idx]
labels_test = labels[test_idx]

import pickle

dataset_dict = {"X_train": X_train, "X_test": X_test, "X_val": X_val, "labels_train": labels_train, "labels_test": labels_test, "labels_val": labels_val}

with open('noveldataset_dict.pickle', 'wb') as file:
    pickle.dump(dataset_dict, file)

with open('noveldataset_dict.pickle', 'rb') as file:
    dataset_dict = pickle.load(file)