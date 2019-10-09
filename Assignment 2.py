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
        self.P_xjyi = sp.sparse.csr_matrix(np.empty([totalWords,totalSubreddits]))


        #Calculate P(yi) for unbalanced data
        countsY = np.unique(TrainingY,return_counts=True)[1] #counts number of occurences of each subreddit (balanced in this case)
        for sub in range(0,totalSubreddits):
            self.P_yi[sub] = countsY[sub]/totalExamples #calculating P(yi), not necessary if data is balanced


        #Calculate P(xj|yi)
        for word in range(0, totalWords): #iterate through all possible words
            if (word%100 == 0): print(word)
            for ex in np.nonzero(TrainingX[:,word])[0]: #iterate through all examples that contain the word
                print(ex)
                subredditIndex = TrainingY[ex] #target of this example
                self.P_xjyi[word][subredditIndex] += 1 #increase count for number of examples with this word for this subreddit









    # def predict(self, InputX):
    #     return predicted


def evaluate_acc(TrueY, PredictedY):
    numCorrect = np.sum(TrueY == PredictedY) #Computes number of correct binary predictions
    percentCorrect = numCorrect / TrueY.shape[0] *100 #Computes percentage of correct predictions

    # print(numCorrect, 'out of',TrueY.shape[0], 'outputs are correct. Percentage correct is',percentCorrect,'%.')

    return percentCorrect



def cross_val_lda(datasetX, datasetY, model, folds):
    splitX = np.array_split(datasetX, folds)
    splitY = np.array_split(datasetY, folds)
    total = 0

    for i in range(0, folds):
        x = splitX[:i] + splitX[i + 1:]
        y = splitY[:i] + splitY[i + 1:]
        x = np.concatenate(x)
        y = np.concatenate(y)
        start = time.time()
        model.fit(x, y)
        end = time.time()
        total += evaluate_acc(splitY[i], model.predict(splitX[i]))
        # print('Runtime: ', end - start)

    print('Average Percentage Correct: ', total / folds)
    return total / folds


if __name__ == '__main__':
    reddit_train = p.read_csv('C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Assignment 2/reddit_train.csv', header=0,)
    reddit_test = p.read_csv('C:/Users/Hansen/Desktop/!School documents/McGill/U4/COMP551/Assignment 2/reddit_test.csv', header=0,)

    reddit_train = reddit_train.to_numpy() #column 0 is id, column 1 is comment words, column 2 is subreddit


    #------------Prepping TrainingY Data
    # encode the subreddit targets
    lb = sklearn.preprocessing.LabelEncoder()
    lb.fit(reddit_train[:,2])
    TrainingY = lb.transform(reddit_train[:,2]) #subreddits in index form 0-19. call lb.classes_[index] to get subreddit name

    #-----------Prepping TrainingX Data
    #vectorize word counts (normal vectorizer)
    vectorizer = CountVectorizer()
    TrainingX = vectorizer.fit_transform(reddit_train[:,1]) #stores vectorized feature data
    #tf-idf vectorizer
    tf_vectorizer = TfidfVectorizer()
    TrainingX = tf_vectorizer.fit_transform(reddit_train[:,1])
    ## normalizing the word counts
    # TrainingX = sklearn.preprocessing.normalize(TrainingX)

    naiveBayesModel = BernoulliNaiveBayes()
    naiveBayesModel.fit(TrainingX,TrainingY)
