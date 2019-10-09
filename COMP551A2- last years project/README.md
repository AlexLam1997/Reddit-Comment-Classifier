# COMP551A2

This code reproduces the results in the report and the models submitted to the Kaggle competition. 

### Prerequisites
The req.txt file was generated using the conda list -e > req.txt command and lists all the required packages for this project. 
For the code to run, the following directories must be present with the data in them. The data is in the same structure in which it was provided, and put in the data folder. 

data/train/pos
data/train/neg
data/test

### Reproducing the Kaggle results
Running predict_test_data.py in the parent directory will save a "foo.csv" file with the predictions. The headers were inserted manually. This file runs grid search for a certain range of C's for Linear SVM, then does the prediction on the test set. 

### Reproducing the figures in the report
The SVM C curve comparison was generated using svm_comparison_vary_features.py
Some line commenting/uncommenting is required to switch between the four different subfigure plots. 

The svm vs logistic regression comparison was generated using svm_vs_logistic_regression.py
Running the file will generate the figure. 

### Naive Bayes from scratch
Running "naive_bayes.py" will create a Naive Bayes model using the training data (must be located as mentioned above). The model will then predict the test data (again must be located as mentioned above) and create and store the predictions in a "naive_bayes_predictions.csv" file. To see the accuracy of the model on the training data, remove the #'s from line 112 and 113.