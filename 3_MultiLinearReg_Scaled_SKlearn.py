# -*- coding: utf-8 -*-
"""
@author: Pachimatla Rajesh

Multiple Linear Regression with feature scaling

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

# Let's explore the top 10 rows of the df
data.head(10)

# There are two independent variables: 'SAT' and 'Rand 1,2,3'
x = data[['SAT','Rand 1,2,3']]

# and a single dependent variable: 'GPA'
y = data['GPA']

#Stardisation
# Import the preprocessing module
# StandardScaler is one of the easiest and 'cleanest' ways to preprocess your data
from sklearn.preprocessing import StandardScaler

# Create an instance of the StandardScaler class
scaler = StandardScaler()

# Fit the input data (x)
# Essentially we are calculating the mean and standard deviation feature-wise 
# (the mean of 'SAT' and the standard deviation of 'SAT', 
# as well as the mean of 'Rand 1,2,3' and the standard deviation of 'Rand 1,2,3')
scaler.fit(x)

# The actual scaling of the data is done through the method 'transform()'
# Let's store it in a new variable, named appropriately
x_scaled = scaler.transform(x)

# Creating a regression works in the exact same way
reg = LinearRegression()

# We just need to specify that our inputs are the 'scaled inputs'
reg.fit(x_scaled,y)

# Let's see the coefficients
print(reg.coef_)
# And the intercept
print(reg.intercept_)


# As usual we can try to arrange the information in a summary table
# Let's create a new data frame with the names of the features
reg_summary = pd.DataFrame(data=[['Bias'],['SAT'],['Rand 1,2,3']], columns=['Features'])
print(reg_summary)
# Then we create and fill a second column, called 'Weights' with the coefficients of the regression
# Since the standardized coefficients are called 'weights' in ML, this is a much better word choice for our case
# Note that even non-standardized coeff. are called 'weights' 
# but more often than not, when doing ML we perform some sort of scaling
reg_summary['Weights'] = [reg.intercept_, reg.coef_[0], reg.coef_[1]]
print(reg_summary)

# For simplicity, let's crete a new dataframe with 2 *new* observations
new_data = pd.DataFrame(data=[[1700,2],[1800,1]],columns=['SAT','Rand 1,2,3'])
print(new_data)

# We can make a prediction for a whole dataframe (not a single value)
# Note that the output is very strange (different from mine)
reg.predict(new_data)
# Our model is expecting SCALED features (features of different magnitude)
# In fact we must transform the 'new data' in the same way as we transformed the inputs we train the model on
# Luckily for us, this information is contained in the 'scaler' object
# We simply transform the 'new data' using the relevant method
new_data_scaled = scaler.transform(new_data)

# Let's check the result
new_data_scaled
# Finally we make a prediction using the scaled new data
reg.predict(new_data_scaled)
# The output is much more appropriate, isn't it?

# Theory suggests that features with very small weights could be removed and the results should be identical
# Moreover, we proved in 2-3 different ways that 'Rand 1,2,3' is an irrelevant feature
# Let's create a simple linear regression (simple, because there is a single feature) without 'Rand 1,2,3'
reg_simple = LinearRegression()

# Once more, we must reshape the inputs into a matrix, otherwise we will get a compatibility error 
# Note that instead of standardizing again, I'll simply take only the first column of x
print(x_scaled)
x_simple_matrix = x_scaled[:,0].reshape(-1,1)
#.reshape(-1, 1) reshapes the extracted column into a new 2D array with a single column. The -1 in the reshape function is a placeholder that means "whatever is necessary to make this work." 

# Finally, we fit the regression
reg_simple.fit(x_simple_matrix,y)

# In a similar manner to the cell before, we can predict only the first column of the scaled 'new data'
# Note that we also reshape it to be exactly the same as x
reg_simple.predict(new_data_scaled[:,0].reshape(-1,1))

##### Hence, feature scaling is the best way to apenalise the variables which are
# not significant #############
