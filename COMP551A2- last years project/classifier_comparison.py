#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:05:39 2019

Comparison of classifiers
Basically we do 
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
applied to our case
@author: vassili
"""

from extract_features import load_raw_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


corpus, reviews=load_raw_data(num_files_to_read_per_sent=12500) #HOW MANY FILES TO LOAD


c_vect = TfidfVectorizer().fit(corpus)
X=c_vect.transform(corpus)
Y=reviews

pclf_glob = Pipeline([ #create sequence of transforms and classifier
    ('norm', Normalizer()), #comment to try without normalization
    ('clf', LinearSVC())
])

names=["MNB", "LinSVM", "DecTree", "RandForest", "LogReg", "AdaBoost"]
classifiers=[
        MultinomialNB(),
        LinearSVC(C=0.25918),
        DecisionTreeClassifier(),
        RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
        LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),
        AdaBoostClassifier()                   
        ]

param_grid = {'clf': classifiers}

pclf=GridSearchCV(pclf_glob, param_grid, cv=4, verbose=1)
pclf.fit(X, Y)
max_idx=np.argmax(pclf.cv_results_['mean_test_score'])

scores=pclf.cv_results_['mean_test_score']
max_idx=np.argmax(pclf.cv_results_['mean_test_score'])


index = np.arange(len(names))
bar_width = 0.35

fig = plt.figure()
ax = plt.gca()
rects = ax.bar(index, scores, bar_width, color='b')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Model')
ax.set_xticks(index)
ax.set_xticklabels(names)
plt.savefig('AllModelComparison.pdf', bbox_inches='tight')

































    

    
    
    
    
    
    
    
    
    
    
    
    
    
    









