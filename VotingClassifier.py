import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd

import training as t

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')


df = pd.read_csv('reddit_train.csv', index_col = 'id')
# Categorize variables, keep mappings to labels

df['category_id'] = df['subreddits'].factorize()[0]
category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
df.head()
labels = df.category_id

features = t.getFeatures()

selectedFeatures = []
selectedFeatures = selectedFeatures + [features[8]]

vectorizedFeatures, vectorizers = t.customVectorize(selectedFeatures)


eclf1 = eclf1.fit(vectorizedFeatures[0:10000], labels[0:10000])
print(eclf1.score(vectorizedFeatures, labels))



