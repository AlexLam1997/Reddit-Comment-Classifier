# -*- coding: utf-8 -*-
import training as t
import pandas as pd
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

features = t.getFeatures()
features = features[4]
features, vectorizers = t.customVectorize([features])
features = features[0]

df = pd.read_csv('reddit_train.csv', index_col = 'id')
# Categorize variables, keep mappings to labels

df['category_id'] = df['subreddits'].factorize()[0]
category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
df.head()
labels = df.category_id

M1 = naive_bayes.MultinomialNB(0.11)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
M1.fit(X_train, y_train)
y_pred = M1.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()