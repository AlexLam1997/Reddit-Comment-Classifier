#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:05:37 2019

@author: artsiomskliar
"""
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob


'''
This function removes makes all stop words as lower and removes stop words
'''
def get_processed_comment(comment, custom_stop_words = []):
    
    stop_words = set(stopwords.words('english'))

    #Get lower case list of words
    text_list = word_tokenize(comment.lower())
    
    #Second pass through after lemmatization is necessary
    text_list = [word for word in text_list
                 if word not in stop_words and word not in custom_stop_words]
    
    return " ".join(text_list)

'''Removes links in comment'''
def get_comment_without_links(comment):
    return re.sub(r"http\S+", "", comment)
    
'''Remove non alphanumeric characters'''
def get_comment_no_punctuation(comment):
    return re.sub("(\\d|\\W)+"," ", comment)
 
'''Returns lemmatized comment'''    
def get_lemmatized_comment(comment, custom_stop_words = []):
    stop_words = set(stopwords.words('english'))

    #Get lower case list of words
    text_list = word_tokenize(comment.lower())
    
    wn = WordNetLemmatizer()
    text_list = [ wn.lemmatize(word, pos="v") for word in text_list 
                 if word not in stop_words and word not in custom_stop_words ]
    
    #Second pass through after lemmatization is necessary
    text_list = [word for word in text_list
                 if word not in stop_words and word not in custom_stop_words]
    
    return " ".join(text_list)

#Adding features using TextBlob

def get_polarity(comment):
    return TextBlob(comment).sentiment[0]

def get_subjectivity(comment):
    return TextBlob(comment).sentiment[1]

def get_num_words(comment):
    words = comment.split()
    words_set = set(words)
    if len(words) > 1:
        return len(words_set)
    else:
        return 0

# Appends desired feature to dataframe
def add_feature(comment_processing_function, current_df, target_column, new_col_name, remove_org_col = False):
    col = current_df[target_column].values
    result_col = list(map(comment_processing_function, col))
    if remove_org_col:
        current_df.drop(columns = [target_column], inplace = True)
    current_df[new_col_name] = result_col
    return current_df
    
