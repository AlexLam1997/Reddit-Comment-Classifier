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

X_no_links = add_feature(get_comment_without_links, df_X.copy(), 'comments', 'comments', remove_org_col = True)
X_no_links.to_csv('./data/X_no_links.csv')

X_processed_no_links = add_feature(get_processed_comment, X_no_links.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_links.to_csv('./data/X_processed_no_links.csv')

X_processed_no_punctuation = add_feature(get_comment_no_punctuation, X_processed_no_links.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_punctuation.to_csv('./data/X_processed_no_punctuation.csv')

X_processed_lemmatized = add_feature(get_lemmatized_comment, X_processed_no_links.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_lemmatized.to_csv('./data/X_processed_lemmatized.csv')

X_processed_no_links_lemmatized = add_feature(get_lemmatized_comment, X_no_links.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_links_lemmatized.to_csv('./data/X_processed_no_links_lemmatized.csv')

X_processed_no_punctuation_lemmatized = add_feature(get_lemmatized_comment, X_processed_no_links.copy(), 'comments', 'comments', remove_org_col = True)
X_processed_no_punctuation_lemmatized.to_csv('./data/X_processed_no_punctuation_lemmatized.csv')

# Finally, have a set with all preprocessing applied
X_all = X_processed_no_links.copy()
X_all = add_feature(get_comment_without_links, X_all, 'comments', 'comments', remove_org_col = True)
X_all = add_feature(get_comment_no_punctuation, X_all, 'comments', 'comments', remove_org_col = True)
X_all.to_csv('./data/X_all.csv')

X_all_lemmatized = add_feature(get_lemmatized_comment, df_X.copy(), 'comments', 'comments', remove_org_col = True)
X_all_lemmatized = add_feature(get_comment_without_links, X_all_lemmatized, 'comments', 'comments', remove_org_col = True)
X_all_lemmatized = add_feature(get_comment_no_punctuation, X_all_lemmatized, 'comments', 'comments', remove_org_col = True)
X_all_lemmatized.to_csv('./data/X_all_lemmatized.csv')


## LOAD FEATURES FROM ./data HERE
X_processed_no_links = pd.read_csv('./data/X_processed_no_links.csv', index_col = 'id')
X_processed_no_punctuation = pd.read_csv('./data/X_processed_no_punctuation.csv', index_col = 'id')
X_no_links = pd.read_csv('./data/X_no_links.csv', index_col = 'id')
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

polarity = pd.read_csv('./data/polarity.csv', header = None).set_index(0)
polarity.columns= ['polarity']

subjectivity = pd.read_csv('./data/subjectivity.csv', header = None).set_index(0)
subjectivity.columns = ['subjectivity']

num_words = pd.read_csv('./data/num_words.csv', header = None).set_index(0)
num_words.columns = ['num_words']


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizers = []
for f in features:
    tfid_reg = TfidfVectorizer()  
    vectorizers.append(tfid_reg)
    
features_vectorizers = list(zip(features, vectorizers))

vectorizers_custom = []
for f in features:
    tfidf_custom = TfidfVectorizer(sublinear_tf=True, 
                           min_df=5, 
                           norm='l2', 
                           encoding='latin-1', 
                           ngram_range=(1, 2), 
                           stop_words='english')
    vectorizers_custom.append(tfidf_custom)
    
features_vectorizers_custom = list(zip(features, vectorizers_custom))

from scipy.sparse import hstack

features_num_cust = []
index = 0
for f, v_reg in features_vectorizers_custom:
    print(index)
    result_reg = v_reg.fit_transform(f.comments)
    result_reg = hstack((result_reg, np.array(polarity.polarity)[:,None]))
    result_reg = hstack((result_reg, np.array(subjectivity.subjectivity)[:,None]))
    result_reg = hstack((result_reg, np.array(num_words.num_words)[:,None]))
    features_num_cust.append(result_reg)
    index+=1

features_num_arr_cust = [ x.toarray() for x in features_num_cust ]      

import csv
index = 0
for f in features_num_arr_cust:
    with open('./data/vectors/' + str(index) + '_custom' +  '.csv', 'w', newline='') as myfile:
         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
         wr.writerow(f)
    index += 1

'''
# Vectorize and append metadata as well
features_num = []
index = 0
for f, v_reg in features_vectorizers:
    print(index)
    result_reg = v_reg.fit_transform(f.comments)
    result_reg = hstack((result_reg, np.array(polarity.polarity)[:,None]))
    result_reg = hstack((result_reg, np.array(subjectivity.subjectivity)[:,None]))
    result_reg = hstack((result_reg, np.array(num_words.num_words)[:,None]))
    features_num.append(result_reg)
    index+=1
    

features_num_arr = [ x.toarray() for x in features_num ]

features_num_cust = []
index = 0


import csv
index = 0
for f in features_num_arr:
    with open('./data/vectors/' + str(index) + '.csv', 'w', newline='') as myfile:
         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
         wr.writerow(f)
    index += 1
    
features_num_parameters = [ len(x[0]) for x in features_num_arr ]
'''

