# -*- coding: utf-8 -*-
"""
@author:Pachimatla Rajesh

Simple linear regression for prediting GPA from SAT score
-
"""

# We will need NumPy, pandas, matplotlib and seaborn, import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# and of course the actual regression (machine learning) module
from sklearn.linear_model import LinearRegression

# We start by loading the data
data = pd.read_csv('Simple linear regression.csv')

# Let's explore the top 5 rows of the df
data.head()

# There is a single independent variable: 'SAT'
x = data['SAT']
# x is series, so better to convert it to a matrix

# and a single depended variable: 'GPA'
y = data['GPA']

print(x.shape,y.shape)

# In order to feed x to sklearn, it should be a 2D array (a matrix)
# Therefore, we must reshape it 
# Note that this will not be needed when we've got more than 1 feature (as the inputs will be a 2D array by default)

# x_matrix = x.values.reshape(84,1)
x_matrix = x.values.reshape(84,1)

# Check the shape just in case
x_matrix.shape


# We start by creating a linear regression object
reg = LinearRegression()

# The whole learning process boils down to fitting the gression
# Note that the first argument is the independent variable, while the second - the dependent (unlike with StatsModels)
reg.fit(x_matrix,y)

# To get the R-squared in sklearn we must call the appropriate method
reg.score(x_matrix,y)

print(reg.intercept_)
print(reg.coef_)

# To be in line with our knowledge so far, we can create a pandas data frame with several different values of SAT
new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
# We can predict the whole data frame in bulk
# Finally, we can directly store the predictions in a new series of the same dataframe
new_data['Predicted_GPA'] = reg.predict(new_data)

# There are different ways to plot the data - here's the matplotlib code
plt.scatter(x,y)

# Parametrized version of the regression line
yhat = reg.coef_*x_matrix + reg.intercept_

# Non-parametrized version of the regression line
#yhat = 0.0017*x + 0.275

# Plotting the regression line
fig = plt.plot(x,yhat, lw=4, c='orange', label ='regression line')

# Labelling our axes
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

################# End of Script ##################3


















