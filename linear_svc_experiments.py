#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:41:23 2019

@author: artsiomskliar
"""

import numpy as np
import pandas as pd
from data_preprocessing import get_processed_comment, get_polarity 
from data_preprocessing import get_subjectivity, get_num_words, add_feature
from data_preprocessing import get_comment_without_links, get_lemmatized_comment, get_comment_no_punctuation

df = pd.read_csv('reddit_train.csv', index_col = 'id')

# Categorize variables, keep mappings to labels

df['category_id'] = df['subreddits'].factorize()[0]
category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
df.head()


df_X = df['comments'].to_frame()
labels = df.category_id

#Store the datasets variations to memory for later
df_X.to_csv('./data/original_X.csv')
labels.to_csv('./data/labels.csv')

#Compute Polarity, Subjectivity, and Length on original comment
polarity = add_feature(get_polarity, df_X.copy(), 'comments', 'polarity')['polarity']
subjectivity = add_feature(get_subjectivity, df_X.copy(), 'comments', 'subjectivity')['subjectivity']
num_words = add_feature(get_num_words, df_X.copy(), 'comments', 'num_words')['num_words']

polarity.to_csv('./data/polarity.csv')
subjectivity.to_csv('./data/subjectivity.csv')
num_words.to_csv('./data/num_words.csv')

# Add pre-process comments and see which processing imapacts the result the most
# Save all of them to reload them later
X_processed = add_feature(get_processed_comment, df_X.copy(), 'comments', 'comments', remove_org_col = True)
X_processed.to_csv('./data/X_processed.csv')

X_processed_no_punctuation = add_feature(get_comment_no_punctuation, X_processed.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_punctuation.to_csv('./data/X_processed_no_punctuation.csv')

X_processed_no_links = add_feature(get_comment_without_links, X_processed.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_links.to_csv('./data/X_processed_no_links.csv')

X_processed_lemmatized = add_feature(get_lemmatized_comment, X_processed.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_lemmatized.to_csv('./data/X_processed_lemmatized.csv')

X_processed_no_links_lemmatized = add_feature(get_lemmatized_comment, X_processed_no_links.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_links_lemmatized.to_csv('./data/X_processed_no_links_lemmatized.csv')

X_processed_no_punctuation_lemmatized = add_feature(get_lemmatized_comment, X_processed.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_punctuation_lemmatized.to_csv('./data/X_processed_no_punctuation_lemmatized.csv')

# Finally, have a set with all preprocessing applied
X_all = X_processed.copy()
X_all = add_feature(get_comment_without_links, X_all, 'comments', 'comments', remove_org_col = True)
X_all = add_feature(get_comment_no_punctuation, X_all, 'comments', 'comments', remove_org_col = True)
X_all.to_csv('X_all.csv')

X_all_lemmatized = X_all.copy()
X_all_lemmatized = add_feature(get_lemmatized_comment, X_all, 'comments', 'comments', remove_org_col = True)
X_all_lemmatized.to_csv('./data/X_all_lemmatized.csv')


## LOAD MODELS FROM ./data HERE
X_processed = pd.read_csv('./data/X_processed.csv', index_col = 'id')
X_processed_no_punctuation = pd.read_csv('./data/X_processed_no_punctuation.csv', index_col = 'id')
X_processed_no_links = pd.read_csv('./data/X_processed_no_links.csv', index_col = 'id')
X_processed_lemmatized = pd.read_csv('./data/X_processed_lemmatized.csv', index_col = 'id')
X_processed_no_links_lemmatized = pd.read_csv('./data/X_processed_no_links_lemmatized.csv', index_col = 'id')
X_processed_no_punctuation_lemmatized = pd.read_csv('./data/X_processed_no_punctuation_lemmatized.csv', index_col = 'id')
X_all = pd.read_csv('X_all.csv', index_col = 'id')
X_all_lemmatized = pd.read_csv('./data/X_all_lemmatized.csv', index_col = 'id')

features = [df_X, # Number of parameters: 38308
            X_processed, # Number of parameters : 36471
            X_processed_no_punctuation, # Number of parameters: 35820
            X_processed_no_links, # Number of parameters: 36434
            X_all, # Numberof parameters: 35780
            X_processed_lemmatized, # Numberof parameters: 36879
            X_processed_no_links_lemmatized, # 36834
            X_processed_no_punctuation_lemmatized, # 
            X_all_lemmatized] # 

polarity = pd.read_csv('./data/polarity.csv', header = None).set_index(0)
polarity.columns= ['polarity']

subjectivity = pd.read_csv('./data/subjectivity.csv', header = None).set_index(0)
subjectivity.columns = ['subjectivity']

num_words = pd.read_csv('./data/num_words.csv', header = None).set_index(0)
num_words.columns = ['num_words']


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizers = []
for f in features:
    tfid = TfidfVectorizer(sublinear_tf=True, 
                           min_df=5, 
                           norm='l2', 
                           encoding='latin-1', 
                           ngram_range=(1, 2), 
                           stop_words='english')
    
    vectorizers.append(tfid)
    
features_vectorizers = list(zip(features, vectorizers))

features_num = []

for f, v in features_vectorizers:
    result = v.fit_transform(f.comments).toarray()
    features_num.append(result)

# Bundle all models together
features_tfidf_vectors = list(zip(features_vectorizers, features_num))

features_num_parameters = []

for f in features_num:
    features_num_parameters.append(len(f[0]))
    
# After reviewing the number of parameters, we pick the two feature sets
# Original dataset, and X_all_lematized dataset
    
#to_test = [ features_num[0], features_num[-1] ]


# Train models to see if we find any pre-processing operations that help more than others
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

results = []

for f in to_test:
    model = LinearSVC()
    accuracies = cross_val_score(model, f, labels, scoring='accuracy', cv = 2)
    score = sum(accuracies) / len(accuracies)
    results.append((model, score))


X_processed_with_polarity = add_feature(get_polarity, 'comments', 'polarity')
X_processed_with_subjectivity = add_feature(get_subjectivity, X_processed, 'comments', 'subjectivity')
X_processed_with_num_words = add_feature(get_num_words, X_processed, 'comments', 'num_words')

# Build with all features
X_all_features = X_processed.copy()
X_all_features['polarity'] = X_processed_with_polarity['polarity']
X_all_features['subjectivity'] = X_processed_with_subjectivity['subjectivity']
X_all_features['num_words'] = X_processed_with_num_words['num_words']


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

#from sklearn.preprocessing import normalize
#features = normalize(features)

# We are ready to create a model - let's first check what cross_validation will yield
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# Test regular model and check cross validation accuracies first. 
svc_model = LinearSVC()
svc_accuracies = cross_val_score(svc_model, features, labels, scoring='accuracy', cv = 5)

from sklearn.ensemble import RandomForestClassifier
forest_model = RandomForestClassifier(n_estimators = 1000)

rf_accuracies = cross_val_score(forest_model, features, labels, scoring='accuracy', cv = 5)




