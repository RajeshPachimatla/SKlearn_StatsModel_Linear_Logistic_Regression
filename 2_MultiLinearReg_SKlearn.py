# -*- coding: utf-8 -*-
"""

@author: Pachimatla Rajesh

-Multiple Linear regression using SKlearn
-F-statistics for featre selection

"""

# For these lessons we will need NumPy, pandas, matplotlib and seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# and of course the actual regression (machine learning) module
from sklearn.linear_model import LinearRegression

# Load the data from a .csv in the same folder
data = pd.read_csv('Multiple linear regression.csv')
#data.columns
data.describe()
# There are two independent variables: 'SAT' and 'Rand 1,2,3'
x = data[['SAT','Rand 1,2,3']]
#here no need to reshape the x, since it is multi-variable

# and a single depended variable: 'GPA'
y = data['GPA']

reg = LinearRegression()

reg.fit(x,y)

print(reg.intercept_, reg.coef_)

# If we want to find the Adjusted R-squared we can do so by knowing the r2, the # observations, the # features
    r2 = reg.score(x,y)
# Number of observations is the shape along axis 0
n = x.shape[0]
# Number of features (predictors, p) is the shape along axis 1
p = x.shape[1]

# We find the Adjusted R-squared using the formula
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2


# Import the feature selection module from sklearn
# This module allows us to select the most appopriate features for our regression
# There exist many different approaches to feature selection, however, we will use one of the simplest
from sklearn.feature_selection import f_regression

# We will look into: f_regression
# f_regression finds the F-statistics for the *simple* regressions created with each of the independent variables
# In our case, this would mean running a simple linear regression on GPA where SAT is the independent variable
# and a simple linear regression on GPA where Rand 1,2,3 is the indepdent variable
# The limitation of this approach is that it does not take into account the mutual effect of the two features
f_regression(x,y)

# There are two output arrays
# The first one contains the F-statistics for each of the regressions
# The second one contains the p-values of these F-statistics

# Since we are more interested in the latter (p-values), we can just take the second array
F_value = f_regression(x, y)[0]
p_values = f_regression(x,y)[1]
# To be able to quickly evaluate them, we can round the result to 3 digits after the dot
p_values.round(3)

print(x.columns.values)
# Let's create a new data frame with the names of the features
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])

# Then we create and fill a second column, called 'Coefficients' with the coefficients of the regression
reg_summary ['Coefficients'] = reg.coef_
# Finally, we add the p-values we just calculated
reg_summary ['p-values'] = p_values.round(3)
print(reg_summary)

#hence, results shows that SAT is the significant feature

############### End of the Script ##############
