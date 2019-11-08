# Reddit-Comment-Classifier

Multiple approaches to redit comment classification. In this experiment we used data pulled from the popular online forum Reddit and aimed to classify the comments into one of 20 different possible subreddits. The popular python machine learning library SkLearn was used for most of the model implementations in this repository. We also experimented with our own implementation of a Bernouilli Naive Bayes model. Our highest training set accuracy was of 58.271%, achieved with a Multinomial Naive Bayes model. The different models we experimented with to acheieve classification were: Linear SVC,  Multinomial Naive Bayes, SVC(Rbf kernel), BERT (1 epoch) and LTSM. 

Files Description

linear_svc_comment_simple.py - Contains simple implementation of a 
linear SVC model. <br> 

feature_building.py - Building and saving of all features sets after pre-processing
 and vectorizations. <br>
 
training.py - Contains all loading of feature sets, vectorizations, and 
training of classical scikilearn models. 

Bernoulli Naive Bayes model is in the "BernoulliNaiveBayes.py" file. Replace first two lines
of "main" with the correct filepath to the reddit_test and reddit_train csv files.

lstm_model.py - Contains the LSTM model trained using Keras library. 
