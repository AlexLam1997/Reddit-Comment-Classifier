#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:05:37 2019

@author: artsiomskliar
"""
import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob


'''
This function removes links from comments, removes punctuation, and lemmatizes
the comment. Returns the processed string
'''
def get_processed_comment(comment, custom_stop_words = []):
    stop_words = set(stopwords.words('english'))
    #remove links and non alphanumeric characters
    comment = re.sub(r"http\S+", "", comment)
    comment = re.sub("(\\d|\\W)+"," ", comment)
    
    #Get lower case list of words
    text_list = word_tokenize(comment.lower())
    
    #Second pass through after lemmatization is necessary
    text_list = [word for word in text_list
                 if word not in stop_words and word not in custom_stop_words]
    
    return " ".join(text_list)

def get_processed_comment_lemmatized(comment, custom_stop_words = []):
    stop_words = set(stopwords.words('english'))
    #remove links and non alphanumeric characters
    comment = re.sub(r"http\S+", "", comment)
    comment = re.sub("(\\d|\\W)+"," ", comment)
    
    #Get lower case list of words
    text_list = word_tokenize(comment.lower())
    
    #Lemmatize if argument is true
    
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
def add_feature(comment_processing_function, current_df, target_column, new_col_name):
    col = current_df[target_column].values
    result_col = list(map(comment_processing_function, col))
    current_df[new_col_name] = result_col
    return current_df
    
