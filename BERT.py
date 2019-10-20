import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import pearsonr
import math
import time
import itertools
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import naive_bayes
import pickle
import csv
import ktrain
from ktrain import text

def evaluate_acc(TrueY, PredictedY):
    numCorrect = np.sum(TrueY == PredictedY) #Computes number of correct binary predictions
    percentCorrect = numCorrect / TrueY.shape[0] *100 #Computes percentage of correct predictions

    print(numCorrect, 'out of',TrueY.shape[0], 'outputs are correct. Percentage correct is',percentCorrect,'%.')

    return percentCorrect



if __name__ == '__main__':
    reddit_train = p.read_csv('C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Reddit-Comment-Classifier/reddit_train.csv', header=0,)
    reddit_test = p.read_csv('C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Reddit-Comment-Classifier/reddit_test.csv', header=0,)

    reddit_train = reddit_train.to_numpy() #column 0 is id, column 1 is comment words, column 2 is subreddit
    reddit_test = reddit_test.to_numpy() #column 0 is id, column 1 is comment words, column 2 is subreddit


    #------------Prepping TrainingY Data
    # encode the subreddit targets
    lb = sklearn.preprocessing.LabelEncoder()
    lb.fit(reddit_train[:,2])
    TrainingY = lb.transform(reddit_train[:,2]) #subreddits in index form 0-19. call lb.classes_[index] to get subreddit name



    #---------------BERT TECHNIQUE------------


    (x_train, y_train),(x_test,y_test), preproc = text.texts_from_array(x_train=reddit_train[:,1].tolist(), y_train=TrainingY.tolist(),preprocess_mode='bert',class_names = reddit_train[:,2].tolist(),maxlen=350)

    model = text.text_classifier(name='bert', train_data=(x_train, y_train), preproc=preproc)
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)

    learner.fit_onecycle(2e-5, 1)

    #Save BERT to pickle
    filename = 'C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Reddit-Comment-Classifier/learner.sav'
    pickle.dump(learner, open(filename, 'wb'))

    filename = 'C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Reddit-Comment-Classifier/preproc.sav'
    pickle.dump(preproc, open(filename, 'wb'))

    filename = 'C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Reddit-Comment-Classifier/model.sav'
    pickle.dump(model, open(filename, 'wb'))


    #---Predicting results
    predictor = ktrain.get_predictor(learner.model, preproc)
    testResults = predictor.predict(reddit_test[:,1].tolist())

# # ----------Writing test results to csv file----------

numTestExamples = TestX.shape[0]
TestX[:, 0]  # gives the ids
testResultsString = [None for _ in range(numTestExamples)]
for ex in range(numTestExamples):
    testResultsString[ex] = lb.classes_[int(testResults[ex])]
with open('testResults.csv', mode='w', newline='') as testResultsf:
    results_writer = csv.writer(testResultsf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Id', 'Category'])
    for ex in range(numTestExamples):
        results_writer.writerow([reddit_test[ex, 0], testResultsString[ex]])



