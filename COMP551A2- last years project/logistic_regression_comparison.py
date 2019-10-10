#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:50:16 2019

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



from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt


corpus, reviews=load_raw_data(num_files_to_read_per_sent=12500) #HOW MANY FILES TO LOAD
X,Y=corpus, reviews

lemma_count = CountVectorizer(tokenizer=MyTokenizer())
lemma_tfidf = TfidfVectorizer(tokenizer=MyTokenizer())

names=["Count", "TFIDF", "CountLemm", "TFIDF_Lemm"]
vectorizers=[CountVectorizer(), TfidfVectorizer(), lemma_count, lemma_tfidf]
pclf_glob = Pipeline([ #create sequence of transforms and classifier
    ('norm', Normalizer()), #comment to try without normalization
    ('clf', LogisticRegression())
])

def evaluate_vectorizer(vectorizer):
    c_vect = vectorizer.fit(corpus)
    X=c_vect.transform(corpus)
    Y=reviews
    scores=cross_val_score(pclf_glob, X, Y, cv=4)
    return np.mean(scores)

vec_scores=np.zeros(len(vectorizers))


for i, vec in enumerate(vectorizers):
    vec_scores[i]=evaluate_vectorizer(vectorizers[i])

index = np.arange(len(names))
bar_width = 0.5

fig = plt.figure()
ax = plt.gca()
rects = ax.bar(index, vec_scores, bar_width, color='b')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Feature Extraction method')
ax.set_xticks(index)
ax.set_xticklabels(names)
plt.savefig('LogisticRegression_Comparison.pdf', bbox_inches='tight')
