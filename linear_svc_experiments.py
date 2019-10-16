#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:41:23 2019

@author: artsiomskliar
"""

import numpy as np
import pandas as pd
from data_preprocessing import get_processed_comment, get_polarity, get_subjectivity, get_num_words, add_feature

df = pd.read_csv('reddit_train.csv', index_col = 'id')

# Categorize variables, keep mappings to labels

df['category_id'] = df['subreddits'].factorize()[0]
category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
df.head()


df_X = df['comments'].to_frame()
df_y = df['category_id'].to_frame()

# Add features such as polarity, subjectivity, and length of comment
df_X = add_feature(get_processed_comment, df_X, 'comments', 'processed_comments')
df_X = add_feature(get_polarity, df_X, 'processed_comments', 'polarity')
df_X = add_feature(get_subjectivity, df_X, 'processed_comments', 'subjectivity')
df_X = add_feature(get_num_words, df_X, 'processed_comments', 'num_words')

#Remove old comments column
df_X.drop(columns='comments', inplace = True)

# Now, apply tfidf onto the columns variable
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

#Process comments
features = tfidf.fit_transform(df_X.processed_comments)

# Add other computed features to feature matrix
from scipy.sparse import hstack
features = hstack((features, np.array(df_X['polarity'])[:, None]))
features = hstack((features, np.array(df_X['subjectivity'])[:, None]))
features = hstack((features, np.array(df_X['num_words'])[:, None]))

from sklearn.preprocessing import normalize
features = normalize(features)

labels = df.category_id

# We are ready to create a model - let's first check what cross_validation will yield
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# Test regular model and check cross validation accuracies first. 
svc_model = LinearSVC()
svc_accuracies = cross_val_score(svc_model, features, labels, scoring='accuracy', cv = 5)

from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(n_estimators = 1000)

rf_accuracies = cross_val_score(forest_model, features, labels, scoring='accuracy', cv = 5)




