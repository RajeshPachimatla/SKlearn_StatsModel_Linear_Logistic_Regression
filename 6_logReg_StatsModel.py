# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

-Logistic Regression example Statsmodel

"""
### import necessary library files ###
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

#Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#The bove code is replacing the chisqprob function with a custom lambda function 
#that calculates the survival function of the chi-squared distribution 
#(i.e., 1 - CDF) using stats.chi2.sf(chisq, df).

raw_data = pd.read_csv('Admittance.csv')
data = raw_data.copy()
data['Admitted'] = raw_data['Admitted'].map({'Yes': 1, 'No': 0})
data

y = data['Admitted']
x1 = data['SAT']

x = sm.add_constant(x1)
#add constant to enter it as matrix, since x1 is 1D
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

# Get the regression summary
results_log.summary()

#Looking into the Null
# Create a variable only of 1s
const = np.ones(168)
const

reg_null = sm.Logit(y,const)
results_null = reg_null.fit()
results_null.summary()

#Plot logistics curves
# Creating a logit regression (we will discuss this in another notebook)
reg_log = sm.Logit(y,x)
# Fitting the regression
results_log = reg_log.fit()

# Creating a logit function, depending on the input and coefficients
def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

# Sorting the y and x, so we can plot the curve
f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))
ax = plt.scatter(x1,y,color='C0')
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('Admitted', fontsize = 20)
# Plotting the curve
ax2 = plt.plot(x_sorted,f_sorted,color='red')
plt.figure(figsize=(20,20))
plt.show()

np.exp(4.20)

######### END #############