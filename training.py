#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Oct 19 15:43:16 2019

@author: artsiomskliar
"""

'''DESCRIPTION - PLEASE READ

This file loads from memory all the comments after pre-processing but prior to vectorization.
It turns out, vectorizing the comments is faster than loading vectors from memory.

IMPORTANT DATASTRUCTURES FOR TRAINING:
    
 features_vectorizers holds an array of 2D-tuples :
     index 0 holds the feature set in text form
     index 1 holds the TfidfVectorizer() object to vectorize it.
     
 features_num_arr[i] holds the vectors for features_vectorizers[i][0]
     
 features_vectorizers_custom is the same, but we used a custom tfidf setting
 
 features_num_arr_cust[i] holds the vector for features_vectorizers_custom[i][0]
 
 Pass the vectors in features_num_arr or features_num_arr_cust to models for training.
 
 To make predicitons, we must use the same TfidfVectorizer() to transform the
 the test data that we used for training. 
 
 Example: if I want to train a model on X_processed, held at index 1,
 (refer to `features` list for index) I will need to used the TfidfVectorizer()
 features_vectorizers[1][1] to transform the new data to predict by using
 X_test_vector = features_vectorizers[1][1].transform(X_test)
 
 Then we can use y_pred = model.predict(X_test_vector) to get predictions
 
 Labels are all stored in `labels` for training set are used in `labels`. 
 
 To obtain translation, use `id_to_category` dictionary to get answer. 

 E.G. answer = id_to_category[y_pred[0]]
 output: 'hockey' if the predicted value was hockey
 
 
 Approximate time for vectorization completion of all 
 features sets : 2 min 36 seconds

'''

import numpy as np
import pandas as pd
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def getFeatures():
    df = pd.read_csv('reddit_train.csv', index_col = 'id')
    df_X = df['comments'].to_frame()
    X_no_links = pd.read_csv('./data/X_no_links.csv', index_col = 'id')
    X_processed_no_links = pd.read_csv('./data/X_processed_no_links.csv', index_col = 'id')
    X_processed_no_punctuation = pd.read_csv('./data/X_processed_no_punctuation.csv', index_col = 'id')
    X_processed_lemmatized = pd.read_csv('./data/X_processed_lemmatized.csv', index_col = 'id')
    X_processed_no_links_lemmatized = pd.read_csv('./data/X_processed_no_links_lemmatized.csv', index_col = 'id')
    X_processed_no_punctuation_lemmatized = pd.read_csv('./data/X_processed_no_punctuation_lemmatized.csv', index_col = 'id')
    X_all = pd.read_csv('./data/X_all.csv', index_col = 'id')
    X_all_lemmatized = pd.read_csv('./data/X_all_lemmatized.csv', index_col = 'id')

    features = [df_X, # Number of parameters: 38308
                X_processed_no_links, # Number of parameters : 36471
                X_processed_no_punctuation, # Number of parameters: 35820
                X_no_links, # Number of parameters: 36434
                X_all, # Numberof parameters: 35780
                X_processed_lemmatized, # Numberof parameters: 36879
                X_processed_no_links_lemmatized, # 36834
                X_processed_no_punctuation_lemmatized, # 
                X_all_lemmatized] # 
    return features

def getAdditionalFeatures():
    polarity = pd.read_csv('./data/polarity.csv', header = None).set_index(0)
    polarity.columns= ['polarity']
    
    subjectivity = pd.read_csv('./data/subjectivity.csv', header = None).set_index(0)
    subjectivity.columns = ['subjectivity']
    
    num_words = pd.read_csv('./data/num_words.csv', header = None).set_index(0)
    num_words.columns = ['num_words']
    return polarity, subjectivity, num_words

def vectorizeData(features, vectorizerTemplate, polarity = None, subjectivity = None, num_words = None):
    """
    vectorizes the passed in features, takes in the features and the vectorizer you want to use
    caller of this function needs to create the vectorizer to be used beforehand
    this function will create copies of the vectorizer to fit transform each feature 
    returns: tuple(array of vectorized features, tuple of vectorizers used)
    """
    vectorizers = []
    for f in features:
        vectorizers.append(copy.deepcopy(vectorizerTemplate))
        
    features_vectorizers = list(zip(features, vectorizers))
    
    # Vectorize and append metadata as well
    features_num = []
    index = 0
    for f, v_reg in features_vectorizers:
        print(index)
        result_reg = v_reg.fit_transform(f.comments)
        if polarity is not None:
            result_reg = hstack((result_reg, np.array(polarity.polarity)[:,None]))
        if subjectivity is not None:
            result_reg = hstack((result_reg, np.array(subjectivity.subjectivity)[:,None]))
        if num_words is not None:
            result_reg = hstack((result_reg, np.array(num_words.num_words)[:,None]))
        features_num.append(result_reg)
        index+=1
        
    # features_num_arr = [ x.todense() for x in features_num ]
    return features_num, features_vectorizers
    
def defaultVectorize(features, polarity = None, subjectivity = None, num_words = None):
    """
    helper method to vectorize features using out of the box vectorizer
    """
    vectorizer = TfidfVectorizer()
    return vectorizeData(features, vectorizer, polarity= polarity, subjectivity = subjectivity, num_words= num_words)

def customVectorize(features, polarity = None, subjectivity = None, num_words = None):
    """
    helper method to vectorize features using our custom vectorizer
    """
    vectorizer = TfidfVectorizer(sublinear_tf=True, 
                           min_df=5, 
                           norm='l2', 
                           encoding='latin-1', 
                           ngram_range=(1, 2), 
                           stop_words='english')
    return vectorizeData(features, vectorizer, polarity= polarity, subjectivity = subjectivity, num_words= num_words)

print('hi')

if __name__ == "__main__":
    features = getFeatures()
    polarity, subjectivity, num_words = getAdditionalFeatures()
    vectorizedfeatures, feature_vectorizers = customVectorize(features)
#
#df = pd.read_csv('reddit_train.csv', index_col = 'id')
## Categorize variables, keep mappings to labels
#
#df['category_id'] = df['subreddits'].factorize()[0]
#category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
#category_to_id = dict(category_id_df.values)
#id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
#df.head()
#labels = df.category_id
#
#
#df_X = df['comments'].to_frame()
#
### LOAD FEATURES FROM ./data HERE
#X_processed = pd.read_csv('./data/X_processed.csv', index_col = 'id')
#X_processed_no_punctuation = pd.read_csv('./data/X_processed_no_punctuation.csv', index_col = 'id')
#X_processed_no_links = pd.read_csv('./data/X_processed_no_links.csv', index_col = 'id')
#X_processed_lemmatized = pd.read_csv('./data/X_processed_lemmatized.csv', index_col = 'id')
#X_processed_no_links_lemmatized = pd.read_csv('./data/X_processed_no_links_lemmatized.csv', index_col = 'id')
#X_processed_no_punctuation_lemmatized = pd.read_csv('./data/X_processed_no_punctuation_lemmatized.csv', index_col = 'id')
#X_all = pd.read_csv('./data/X_all.csv', index_col = 'id')
#X_all_lemmatized = pd.read_csv('./data/X_all_lemmatized.csv', index_col = 'id')
#
#features = [df_X, # Number of parameters: 
#            X_processed, # Number of parameters : 
#            X_processed_no_punctuation, # Number of parameters: 
#            X_processed_no_links, # Number of parameters: 
#            X_all, # Numberof parameters: 
#            X_processed_lemmatized, # Numberof parameters: 
#            X_processed_no_links_lemmatized, # 
#            X_processed_no_punctuation_lemmatized, # 
#            X_all_lemmatized] # 
#
#polarity = pd.read_csv('./data/polarity.csv', header = None).set_index(0)
#polarity.columns= ['polarity']
#
#subjectivity = pd.read_csv('./data/subjectivity.csv', header = None).set_index(0)
#subjectivity.columns = ['subjectivity']
#
#num_words = pd.read_csv('./data/num_words.csv', header = None).set_index(0)
#num_words.columns = ['num_words']
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#from scipy.sparse import hstack
#
## out of the box vectorising of data
#vectorizers = []
#for f in features:
#    tfid_reg = TfidfVectorizer()  
#    vectorizers.append(tfid_reg)
#    
#features_vectorizers = list(zip(features, vectorizers))
#
## Vectorize and append metadata as well
#features_num = []
#index = 0
#for f, v_reg in features_vectorizers:
#    print(index)
#    result_reg = v_reg.fit_transform(f.comments)
#    result_reg = hstack((result_reg, np.array(polarity.polarity)[:,None]))
#    result_reg = hstack((result_reg, np.array(subjectivity.subjectivity)[:,None]))
#    result_reg = hstack((result_reg, np.array(num_words.num_words)[:,None]))
#    features_num.append(result_reg)
#    index+=1
#    
#features_num_arr = [ x.toarray() for x in features_num ]
#
#
#
## custom vectorizing of data
#vectorizers_custom = []
#for f in features:
#    tfidf_custom = TfidfVectorizer(sublinear_tf=True, 
#                           min_df=5, 
#                           norm='l2', 
#                           encoding='latin-1', 
#                           ngram_range=(1, 2), 
#                           stop_words='english')
#    vectorizers_custom.append(tfidf_custom)
#    
#features_vectorizers_custom = list(zip(features, vectorizers_custom))
#
#features_num_cust = []
#index = 0
#for f, v_reg in features_vectorizers_custom:
#    print(index)
#    result_reg = v_reg.fit_transform(f.comments)
#    result_reg = hstack((result_reg, np.array(polarity.polarity)[:,None]))
#    result_reg = hstack((result_reg, np.array(subjectivity.subjectivity)[:,None]))
#    result_reg = hstack((result_reg, np.array(num_words.num_words)[:,None]))
#    features_num_cust.append(result_reg)
#    index+=1
#
#features_num_arr_cust = [ x.toarray() for x in features_num_cust ]   



# Train models to see if we find any pre-processing operations that help more than others
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
