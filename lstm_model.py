#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:30:13 2019

@author: artsiomskliar
"""

import numpy as np
import pandas as pd
from data_preprocessing import get_processed_comment
from data_preprocessing import get_comment_no_punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout




df = pd.read_csv('reddit_train.csv', index_col = 'id')

# Categorize variables, keep mappings to labels

df['category_id'] = df['subreddits'].factorize()[0]
category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
df.head()


df_X = df['comments'].to_frame()
labels = df.category_id

## LOAD FEATURES FROM ./data HERE
X_processed = pd.read_csv('./data/X_processed.csv', index_col = 'id')
X_processed_no_punctuation = pd.read_csv('./data/X_processed_no_punctuation.csv', index_col = 'id')
X_processed_no_links = pd.read_csv('./data/X_processed_no_links.csv', index_col = 'id')
X_processed_lemmatized = pd.read_csv('./data/X_processed_lemmatized.csv', index_col = 'id')
X_processed_no_links_lemmatized = pd.read_csv('./data/X_processed_no_links_lemmatized.csv', index_col = 'id')
X_processed_no_punctuation_lemmatized = pd.read_csv('./data/X_processed_no_punctuation_lemmatized.csv', index_col = 'id')
X_all = pd.read_csv('./data/X_all.csv', index_col = 'id')
X_all_lemmatized = pd.read_csv('./data/X_all_lemmatized.csv', index_col = 'id')

features = [df_X, # Number of parameters: 
            X_processed, # Number of parameters : 
            X_processed_no_punctuation, # Number of parameters: 
            X_processed_no_links, # Number of parameters: 
            X_all, # Numberof parameters: 
            X_processed_lemmatized, # Numberof parameters: 
            X_processed_no_links_lemmatized, # 
            X_processed_no_punctuation_lemmatized, # 
            X_all_lemmatized] # 

# We pick X_processed to feed to model. This only made every word lower caps
# and removed english stop words

MAX_NUM_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100


# Tokenizer object
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS, lower = True)

#Tokenize comments
tokenizer.fit_on_texts(X_processed['comments'].values)
word_index = tokenizer.word_index

#Turn comments into sequences
X = tokenizer.texts_to_sequences(X_processed['comments'].values)

# Pad or truncuate the sequence to make them of equal length
X = pad_sequences(X, maxlen = MAX_SEQUENCE_LENGTH)

# Get integer encodings
Y = pd.get_dummies(df['subreddits']).values

# Split training data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Build Sequential model
model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

from keras.models import load_model

model.save('./lstm_keras_model.h5')

model = load_model('./lstm_keras_model.h5')

labels = df.subreddits.value_counts().index.tolist()

accr = model.evaluate(X_val, Y_val)

