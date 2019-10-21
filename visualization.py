#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:54:44 2019

@author: artsiomskliar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cust_s = '38311    & 36474     & 35823                       & 36437                 & 35783                                 & 36882                   & 36837                             & 36882                                   & 36245'

num_par_default = [72268, 74249, 69835, 74244, 69832, 67526, 67521, 67526, 63127]
num_par_custom = s

x_labels = ['Unprocessed', 
            'Processed', 
            'No Punctuation', 
            'No Links', 
            'No Links \n& No Punctuation',
            'Processed \n& Lemmatized', 
            'Processed, No Links \n& Lemmatized', 
            'Processed, \nNo Punctuation & Lemmatized', 
            'Processed, No Links, \nNo Punctuation & Lemmatized']

fig = plt.figure()
plt.title("Number of Parameters After Vectorization")
plt.xlabel("Feature Set")
plt.xticks(rotation = 'vertical')
plt.ylabel("Number of Parameters")
plt.bar(x_labels, num_par_default)
plt.bar(x_labels, num_par_custom)
plt.legend(['Default Vectorizer', 'Custom Vectorizer'])
plt.gcf().subplots_adjust(bottom=0.5)
plt.show()
fig.savefig('num_param.png')