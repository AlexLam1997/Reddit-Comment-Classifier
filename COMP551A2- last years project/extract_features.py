#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:03:22 2019

@author: vassili
"""

import os
import numpy as np
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class MyTokenizer(object):
    def __init__(self):
        """Copy pasted from the sklearn feature extraction documentation
        https://scikit-learn.org/stable/modules/feature_extraction.html"""
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def load_raw_data(num_files_to_read_per_sent):
    """Load raw files into python lists. Assumes each file is in e.g. data/train/pos
    corpus: list of strings, one for each file
    reviews: vector, 1 if corresponding corpus is pos, 0 if neg.  """
    corpus=[]
    reviews=-1*np.ones([num_files_to_read_per_sent*2])
    i=0;
    for foldername in ['pos', 'neg']:
        file_list=os.listdir('data/train/'+ foldername)
        for fname in file_list[0:num_files_to_read_per_sent]:
            path='data/train/'+foldername+'/'+fname
            f=open(path)
            raw=f.read()
            corpus.append(raw)
            reviews[i]=(1 if foldername=='pos' else 0)
            i=i+1;
    return corpus, reviews

def construct_data_matrices():
    """Was initial version, here only for demosntraion/reference"""
    corpus, reviews=load_raw_data(num_files_to_read_per_sent=100) #HOW MANY FILES TO LOAD
    X_train, X_val, y_train, y_val = train_test_split(corpus, reviews, train_size=0.8, test_size=0.2)
    
    lemma_vect = CountVectorizer(tokenizer=MyTokenizer()) #Use a lemmatizer, add to the countvectorizer
    count_vect = lemma_vect.fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    X_val_counts = count_vect.transform(X_val) 
    
#tf-idf only
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    X_val_tfidf = tfidf_transformer.transform(X_val_counts)
    
    #Both together
    X_train_mat=hstack([X_train_counts, X_train_tfidf]).toarray()
    X_val_mat=hstack([X_val_counts, X_val_tfidf]).toarray()
    return X_train_mat, X_val_mat
















