import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd



clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

X_processed = pd.read_csv('./data/X_processed.csv', index_col = 'id')
X_processed_no_punctuation = pd.read_csv('./data/X_processed_no_punctuation.csv', index_col = 'id')
X_processed_no_links = pd.read_csv('./data/X_processed_no_links.csv', index_col = 'id')
X_processed_lemmatized = pd.read_csv('./data/X_processed_lemmatized.csv', index_col = 'id')
X_processed_no_links_lemmatized = pd.read_csv('./data/X_processed_no_links_lemmatized.csv', index_col = 'id')
X_processed_no_punctuation_lemmatized = pd.read_csv('./data/X_processed_no_punctuation_lemmatized.csv', index_col = 'id')
X_all = pd.read_csv('./data/X_all.csv', index_col = 'id')
X_all_lemmatized = pd.read_csv('./data/X_all_lemmatized.csv', index_col = 'id')