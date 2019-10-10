#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:16:01 2019

@author: vassili
"""
from extract_features import MyTokenizer, load_raw_data
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


from matplotlib import pyplot as plt
corpus, reviews=load_raw_data(num_files_to_read_per_sent=12500) #HOW MANY FILES TO LOAD
X,Y=corpus, reviews

lemma_count = CountVectorizer(tokenizer=MyTokenizer())
lemma_tfidf = TfidfVectorizer(tokenizer=MyTokenizer())

vect_names=["Count", "TFIDF", "CountLemm", "TFIDF_Lemm"]
vectorizers=[CountVectorizer(), TfidfVectorizer(), lemma_count, lemma_tfidf]
pclf_glob = Pipeline([ #create sequence of transforms and classifier
    ('norm', Normalizer()), #comment to try without normalization
    ('clf', LogisticRegression())
])

clf_names=["LogReg", "LinSVC"]
param_grid={'clf':[LogisticRegression(), LinearSVC(C=0.254)]}

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

log_scores=np.zeros(len(vectorizers))
svm_scores=np.zeros(len(vectorizers))

for i, vec in enumerate(vectorizers):
    scores, max_idx=evaluate_vectorizer(vectorizers[i])
    log_scores[i]=scores[0]
    svm_scores[i]=scores[1]

print(log_scores)
print(svm_scores)

names=vect_names

index = np.arange(len(names))
bar_width=0.4

fig = plt.figure()
ax=plt.gca()
log_rects = ax.bar(index, log_scores, bar_width, color='b',label='Logistic')
svm_rects = ax.bar(index+bar_width, svm_scores, bar_width, color='r',label='SVM')

ax.set_ylabel('Accuracy', fontsize=13)
ax.set_xlabel('Vectorizer', fontsize=13)
plt.ylim(0.8, 0.9)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(names, fontsize=13)
ax.legend(fontsize=13, loc='upper left')
plt.savefig('logis_vs_svm.pdf', bbox_inches='tight')
print("Max score SVM:{}".format(np.max(svm_scores)))
print("Max score logistic regression: {}".format(np.max(log_scores)))

















