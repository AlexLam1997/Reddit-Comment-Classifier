#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns

# Import data
df = pd.read_csv('./reddit_train.csv', index_col = 'id')

# Remove any null values
df = df[pd.notnull(df['comments'])]

# Categorize variables, keep mappings to labels

df['category_id'] = df['subreddits'].factorize()[0]
category_id_df = df[['subreddits', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'subreddits']].values)
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.comments).toarray()
labels = df.category_id
features.shape

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# Test regular model and check cross validation accuracies first. 
model = LinearSVC()
accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv = 5)

overall_accuracy = sum(accuracies) / len(accuracies)

# Perform it one last time to check with confusion matrix
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get accuracy score
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)

# Create confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.subreddits.values, yticklabels=category_id_df.subreddits.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Now train model on whole data set and labels
model.fit(features, labels)

# Save this model to file
from joblib import dump, load
dump(model, 'simple_lvc_model.joblib')

#Load model from memory
model = load('simple_lvc_model.joblib')

# Now  test submissions
test_df = pd.read_csv('reddit_test.csv', index_col = 'id')

test_comments = test_df.comments.values.tolist()
text_features = tfidf.transform(test_comments)

test_preds = model.predict(text_features)

# Now translate the predictions back to labels
test_preds_labels = []
for p in test_preds:
    test_preds_labels.append(id_to_category[p])
    
Id = np.arange(0, len(test_preds_labels))

d = {'Id': Id, 'Category': test_preds_labels}
final_df = pd.DataFrame(data=d)

final_df.to_csv('./simple_linear_svc_results.csv', index=False)
