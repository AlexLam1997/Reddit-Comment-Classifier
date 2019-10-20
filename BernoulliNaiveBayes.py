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

##-----------------------------------------------------------------BernoulliNaiveBayes
class BernoulliNaiveBayes(object):
    def __init__(self):
        self.P_yi = np.array([]) #stores probabilty of class yi
        self.P_xjyi = np.array([]) #stores probability of feature xj given class yi


    def fit(self, TrainingX, TrainingY):
        totalExamples = TrainingX.shape[0]
        totalWords = TrainingX.shape[1]

        totalSubreddits = max(TrainingY)+1

        self.P_yi = np.empty(totalSubreddits)
        # self.P_xjyi = sp.sparse.lil_matrix(( totalWords, totalSubreddits ))
        self.P_xjyi = np.empty([totalWords,totalSubreddits])


        #Calculate P(yi) for unbalanced data
        countsY = np.unique(TrainingY,return_counts=True)[1] #counts number of occurences of each subreddit (balanced in this case)
        for sub in range(0,totalSubreddits):
            self.P_yi[sub] = countsY[sub]/totalExamples #calculating P(yi), not necessary if data is balanced


        #Calculate P(xj|yi)
        for word in range(0, totalWords): #iterate through all possible words
            if (word%100 == 0): print(word)
            for ex in np.nonzero(TrainingX[:,word])[0]: #iterate through all examples that contain the word
                subredditIndex = TrainingY[ex] #target of this example
                self.P_xjyi[word,subredditIndex] += 1 #increase count for number of examples with this word for this subreddit

        for col in range(self.P_xjyi.shape[1]):
            self.P_xjyi[:,col]  = (self.P_xjyi[:,col] + 1) / (countsY[col] + 2) #computes P(xj|yi) by dividing by number of examples with the subreddit





    def predict(self,inputX):

        P_xyi = sp.sparse.lil_matrix(( inputX.shape[0], self.P_yi.shape[0] )) #P(x|yi)
        P_yix = sp.sparse.lil_matrix(( inputX.shape[0], self.P_yi.shape[0] )) #P(yi|x) proportional to P(x|yi)*P(yi)
        predicted_yi = np.empty(inputX.shape[0])


        for ex in range(inputX.shape[0]):
            for word in np.nonzero(inputX[ex,:])[1]:
                for sub in range(0,self.P_yi.shape[0]):
                    if P_xyi [ex,sub] == 0 : #for uninitialized matrix values
                        P_xyi [ex,sub] = self.P_xjyi[word,sub] #computes P(x|yi) for each input example
                        # print(P_xyi[ex,sub])
                    else:
                        P_xyi[ex, sub] = P_xyi[ex, sub] * self.P_xjyi[word, sub]
                        # print(P_xyi[ex,sub])


        for sub in range(self.P_yi.shape[0]):
            P_yix[:,sub] = P_xyi[:,sub] * self.P_yi[sub]
            # print(P_yix[:,sub])
            #P_yix stores the probability P(yi|x) for the chance of an input x being from each the subreddits. Take index of maximum value in each row for prediction

        for ex in range(inputX.shape[0]):
            predicted_yi[ex] = np.argmax(P_yix[ex,:].toarray())

        return predicted_yi #returns a list where the index is the example number, and the value is the subreddit



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

    #-----------Prepping TrainingX Data
    #vectorize word counts (normal vectorizer)
    vectorizer = CountVectorizer(max_df=1000,min_df = 3)
    TrainingX = vectorizer.fit_transform(reddit_train[:, 1])  # stores vectorized feature data
    print(len(vectorizer.vocabulary_))
    TestX = vectorizer.transform(reddit_test[:, 1])


 #---------------------BERNOULLI NAIVE BAYES----------------------

    #--------Training the model
    naiveBayesModel = BernoulliNaiveBayes()
    naiveBayesModel.fit(TrainingX,TrainingY)


    #---Save model to pickle
    filename = 'C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Reddit-Comment-Classifier/naiveBayesModel.sav'
    pickle.dump(naiveBayesModel, open(filename, 'wb'))

    #---Loading model from pickle
    with open(filename, 'rb') as f:
        naiveBayesModel = pickle.load(f)


    # #----Obtaining training predictions using our trained model
    trainingResults = naiveBayesModel.predict(TrainingX)
    print("Computing training set accuracy")
    acc = evaluate_acc(TrainingY,trainingResults)


    #---Predicting test results
    testResults = naiveBayesModel.predict(TestX)
    filename = 'testResults.sav'
    pickle.dump(testResults, open(filename, 'wb'))

    #---Loading test results (if necessary)
    with open('testResults.sav', 'rb') as f:
        testResults = pickle.load(f)





# ----------Writing test results to csv file----------

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



