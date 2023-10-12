# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Binary Predictor with two independent variables
-logistic regression using Statsmodel
-logistic regression using SciKit learn

"""
### Import relenant libraries ###
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('Binary_predictors.csv')
raw_data

data = raw_data.copy()
#convert the categorical variables into 1 or 0
data['Admitted'] = data['Admitted'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

y = data['Admitted']
x1 = data[['SAT','Gender']]

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()

#Note: 
#1. log-likelihood value is much higher for SAT+Gender than only Gender
#2. p-value for Gender variable is 0.022 < 0.05, it shows still it is significant

#Accuracy calculations
# This is a method to change the formatting of np arrays
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
# Should you want to go back to the default formatting, uncomment and execute the line below
#np.set_printoptions(formatter=None)
results_log.predict()
# An array containing the TRUE (actual) values
#data['Admitted']
np.array(data['Admitted'])
# A prediction table (confusion matrix) showing the 
results_log.pred_table()
# Some neat formatting to read the table (better when seeing it for the first time)
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df
# Create an array (so it is easier to calculate the accuracy)
cm = np.array(cm_df)
# Calculate the accuracy of the model
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train

## Testing model with accuracy
# Load the test dataset
test = pd.read_csv('Test_dataset.csv')
# Map the test data as you did with the train data
test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No': 0})
test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})
test
# Get the actual values (true valies ; targets)
test_actual = test['Admitted']
# Prepare the test data to be predicted
test_data = test.drop(['Admitted'],axis=1)
test_data = sm.add_constant(test_data)
test_data

def confusion_matrix(data,actual_values,model):
        
        # Confusion matrix 
        
        # Parameters
        # ----------
        # data: data frame or array
            # data is a data frame formatted in the same way as your input data (without the actual values)
            # e.g. const, var1, var2, etc. Order is very important!
        # actual_values: data frame or array
            # These are the actual values from the test_data
            # In the case of a logistic regression, it should be a single column with 0s and 1s
            
        # model: a LogitResults object
            # this is the variable where you have the fitted model 
            # e.g. results_log in this course
        # ----------
        
        #Predict the values using the Logit model
        pred_values = model.predict(data)
        # Specify the bins 
        bins=np.array([0,0.5,1])
        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
        # if they are between 0.5 and 1, they will be considered 1
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        # Calculate the accuracy
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        # Return the confusion matrix and the accuracy
        return cm, accuracy
    
    # Create a confusion matrix with the test data
cm = confusion_matrix(test_data,test_actual,results_log)
cm

# Format for easier understanding (not needed later on)
cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df

# Check the missclassification rate
# Note that Accuracy + Missclassification rate = 1 = 100%
print ('Missclassification rate: '+str((1+1)/19))


### SKLearn Logistic Regression ####
#importing library for logistic regression
from sklearn.linear_model import LogisticRegression

#importing performance metrics - accuracy and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
#make an instance of the model

logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(x1,y)
logistic.coef_
logistic.intercept_

test_x1 = test_data.drop(['const'],axis=1)

#Prediction from test data
prediction = logistic.predict(test_x1)
print(prediction)

#Confusion matrix gives true_pos, true_neg, False_pos, False_neg
confusion_matrix1 = confusion_matrix(test_actual, prediction)
print(confusion_matrix1)

#Calculate the accuracy
accuracy_score1 = accuracy_score(test_actual, prediction)
print(accuracy_score1)

#Printing misclassified values from the predictions
print('misclassied sample number :%d' % (test_actual != prediction).sum())

########### end of Script ##############
