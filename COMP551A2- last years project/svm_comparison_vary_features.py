#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:24:16 2019

@author: vassili

Compare the Linear SVC grid search with different data preprocessing techniques. 
"""

from extract_features import MyTokenizer, load_raw_data
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt

corpus, reviews=load_raw_data(num_files_to_read_per_sent=12500) #HOW MANY FILES TO LOAD

C_arr=np.linspace(0.01, 4, 50)
param_grid = {'clf__C': C_arr.tolist()}

pclf_glob = Pipeline([ #create sequence of transforms and classifier
    #('norm', Normalizer()), #comment to try without normalization
    ('clf', LinearSVC())
])

def evaluate_vectorizer(vectorizer):
    c_vect = vectorizer.fit(corpus)
    X=c_vect.transform(corpus)
    Y=reviews
    
    pclf=GridSearchCV(pclf_glob, param_grid, cv=4, verbose=1)
    pclf.fit(X, Y)
    max_idx=np.argmax(pclf.cv_results_['mean_test_score'])
    
    scores=pclf.cv_results_['mean_test_score']
    max_idx=np.argmax(pclf.cv_results_['mean_test_score'])
    return scores, max_idx

def plot_vectorizer_comparison(feature_extraction_names, vectorizers, figname):
    print("Running{}".format(figname))
    for (name, vectorizer) in zip(feature_extraction_names,vectorizers):
        scores, max_idx=evaluate_vectorizer(vectorizer)
        plt.plot(C_arr, scores, label=name)
        plt.scatter(C_arr[max_idx], scores[max_idx],
                    label=name+" Max Acc C={:10.3f}".format(C_arr[max_idx]))
        
        plt.xlabel("C value", fontsize=13)
        plt.ylabel("Cross Validation Accuracy", fontsize=13)
        plt.legend()
        plt.savefig(figname+'.pdf',bbox_inches='tight')
        

#can also add 
#for lemmatizer comparison.. todo?
names=["TF-IDF", "Count"]
vects=[TfidfVectorizer(), CountVectorizer()]
#plot_vectorizer_comparison(names, vects, 'LinSVM_TFIDFvsCount_NoNormalizer')


lemma_count = CountVectorizer(tokenizer=MyTokenizer())
lemma_tfidf = TfidfVectorizer(tokenizer=MyTokenizer())
names=["TF-IDF", "Count"]
vects=[lemma_tfidf, lemma_count]

plot_vectorizer_comparison(names, vects, 'LinSVM_TFIDFvsCount_Lemma_NoNormalizer')






























